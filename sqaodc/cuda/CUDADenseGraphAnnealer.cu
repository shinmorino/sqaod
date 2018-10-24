#include "CUDADenseGraphAnnealer.h"
#include <sqaodc/common/internal/ShapeChecker.h>
#include "DeviceKernels.h"
#include "devfuncs.cuh"
#include <cub/cub.cuh>
#include "DeviceBatchedDot.cuh"
#include <algorithm>

namespace sqint = sqaod_internal;
using namespace sqaod_cuda;

#define SQAODC_VECTORIZE_JQ

#ifdef SQAODC_VECTORIZE_JQ
template<class real>
using DotJq = DeviceDotJqVec4<real, real*>;
#else
template<class real>
using DotJq = DeviceDotJq<real, real*>;
#endif


template<class real>
CUDADenseGraphAnnealer<real>::CUDADenseGraphAnnealer() {
    devStream_ = NULL;
    m_ = -1;
    d_reachCount_ = NULL;
    selectAlgorithm(sq::algoDefault);
}

template<class real>
CUDADenseGraphAnnealer<real>::CUDADenseGraphAnnealer(Device &device) {
    devStream_ = NULL;
    m_ = -1;
    d_reachCount_ = NULL;
    assignDevice(device);
    selectAlgorithm(sq::algoDefault);
}

template<class real>
CUDADenseGraphAnnealer<real>::~CUDADenseGraphAnnealer() {
    deallocate();
    d_random_.deallocate();
    if (d_reachCount_ != NULL)
        devAlloc_->deallocate(d_reachCount_);
    d_reachCount_ = NULL;
    if (dotJq_ != NULL) {
        delete dotJq_;
        dotJq_ = NULL;
    }
}

template<class real>
void CUDADenseGraphAnnealer<real>::deallocate() {
    deallocateProblem();
    deallocateInternalObjects();
}

template<class real>
void CUDADenseGraphAnnealer<real>::deallocateProblem() {
    devAlloc_->deallocate(d_J_);
    devAlloc_->deallocate(d_h_);
    devAlloc_->deallocate(d_c_);
    clearState(solProblemSet);
}

template<class real>
void CUDADenseGraphAnnealer<real>::deallocateInternalObjects() {
    devAlloc_->deallocate(d_matq_);
    devAlloc_->deallocate(d_Jq_);
        
    HostObjectAllocator halloc;
    halloc.deallocate(h_E_);
    halloc.deallocate(h_q_);
    E_ = HostVector();
        
    flipPosBuffer_.deallocate();
    realNumBuffer_.deallocate();

    clearState(solPrepared);
    clearState(solQSet);
}

template<class real>
void CUDADenseGraphAnnealer<real>::assignDevice(sqaod::cuda::Device &device) {
    assignDevice(static_cast<Device&>(device));
}

template<class real>
void CUDADenseGraphAnnealer<real>::assignDevice(Device &device) {
    throwErrorIf(devStream_ != NULL, "Device assigned more than once.");
    devStream_ = device.defaultStream();
    devAlloc_ = device.objectAllocator();
    devFormulas_.assignDevice(device, devStream_);
    devCopy_.assignDevice(device, devStream_);
    d_random_.assignDevice(device, devStream_);
    flipPosBuffer_.assignDevice(device, devStream_);
    realNumBuffer_.assignDevice(device, devStream_);

    d_reachCount_ = (uint2*)devAlloc_->allocate(sizeof(uint2));

    /* initialize sumJq */
    dotJq_ = new DotJq<real>(device, devStream_);
}

template<class real>
sq::Algorithm CUDADenseGraphAnnealer<real>::selectAlgorithm(sq::Algorithm algo) {
    switch (algo) {
    case sq::algoSANaive:
    case sq::algoSAColoring:
        algo_ = sq::algoSANaive;
        return algo_;
    case sq::algoColoring:
    case sq::algoDefault:
        algo_ = sq::algoColoring;
        return algo_;
    default:
        sq::log("Uknown algo, %s, defaulting to %s.",
                sq::algorithmToString(algo), sq::algorithmToString(sq::algoColoring));
        algo_ = sq::algoColoring;
        return sq::algoColoring;
    }
}

template<class real>
sq::Algorithm CUDADenseGraphAnnealer<real>::getAlgorithm() const {
    return algo_;
}


template<class real>
void CUDADenseGraphAnnealer<real>::seed(unsigned long long seed) {
    throwErrorIf(devStream_ == NULL, "Device not set.");
    d_random_.seed(seed);
    setState(solRandSeedGiven);
}

template<class real>
void CUDADenseGraphAnnealer<real>::setQUBO(const HostMatrix &W, sq::OptimizeMethod om) {
    sqint::matrixCheckIfSymmetric(W, __func__);
    throwErrorIf(devStream_ == NULL, "Device not set.");
    deallocate();

    N_ = W.rows;
    m_ = N_ / 4;
    om_ = om;

    devAlloc_->allocate(&d_J_, N_, N_);
    DeviceMatrix *dW = devStream_->tempDeviceMatrix<real>(W.dim(), __func__);
    devCopy_(dW, W);
    if (om == sq::optMaximize)
        devFormulas_.devMath.scale(dW, -1., *dW);
    devFormulas_.calculateHamiltonian(&d_h_, &d_J_, &d_c_, *dW);
    devCopy_.clearPadding(&d_J_);
    devStream_->synchronize();

    setState(solProblemSet);
}

template<class real>
void CUDADenseGraphAnnealer<real>::setHamiltonian(const HostVector &h, const HostMatrix &J,
                                                  real c) {
    sqint::isingModelShapeCheck(h, J, c, __func__);
    sqint::matrixCheckIfSymmetric(J, __func__);
    throwErrorIf(devStream_ == NULL, "Device not set.");
    deallocate();

    N_ = J.rows;
    m_ = N_ / 4;
    om_ = sq::optMinimize;

    devCopy_(&d_h_, h);
    devCopy_(&d_J_, J);
    devCopy_.clearPadding(&d_J_);
    devCopy_(&d_c_, c);
    devStream_->synchronize();

    setState(solProblemSet);
}

template<class real>
sq::Preferences CUDADenseGraphAnnealer<real>::getPreferences() const {
    sq::Preferences prefs = Base::getPreferences();
    prefs.pushBack(sq::Preference(sq::pnDevice, "cuda"));
    return prefs;
}

template<class real>
const sq::VectorType<real> &CUDADenseGraphAnnealer<real>::get_E() const {
    if (!isEAvailable())
        const_cast<This*>(this)->calculate_E();
    /* add a flag to tell if kernel synchronized.*/
    devStream_->synchronize();
    return E_;
}

template<class real>
const sq::BitSetArray &CUDADenseGraphAnnealer<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return xlist_;
}


template<class real>
void CUDADenseGraphAnnealer<real>::set_q(const BitSet &q) {
    sqint::isingModelSolutionShapeCheck(N_, q, __func__);
    throwErrorIfNotPrepared();
    
    DeviceBitSet *d_q = devStream_->tempDeviceVector<char>(q.size);
    devCopy_(d_q, q);
    devCopy_.broadcastToRows(&d_matq_, *d_q);
    devCopy_.clearPadding(&d_matq_);
    devStream_->synchronize();
    setState(solQSet);
}

template<class real>
void CUDADenseGraphAnnealer<real>::set_qset(const BitSetArray &q) {
    sqint::isingModelSolutionShapeCheck(N_, q, __func__);
    m_ = q.size();
    prepare();

    /* FIXME: apply pinned memory */
    sq::BitMatrix qMat(m_, N_);
    for (int iRow = 0; iRow < m_; ++iRow)
        memcpy(&qMat(iRow, 0), q[iRow].data, sizeof(char) * N_);
    DeviceBitMatrix *d_q = devStream_->tempDeviceMatrix<char>(m_, N_);
    devCopy_(&d_matq_, qMat);
    devCopy_.clearPadding(&d_matq_);
    devCopy_.synchronize();
    
    setState(solQSet);
}


template<class real>
void CUDADenseGraphAnnealer<real>::getHamiltonian(HostVector *h, HostMatrix *J, real *c) const {
    throwErrorIfProblemNotSet();

    devCopy_(h, d_h_);
    devCopy_(J, d_J_);
    devCopy_(c, d_c_);
    devCopy_.synchronize();
}

template<class real>
const sq::BitSetArray &CUDADenseGraphAnnealer<real>::get_q() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return qlist_;
}

template<class real>
void CUDADenseGraphAnnealer<real>::randomizeSpin() {
    throwErrorIfNotPrepared();

    ::randomizeSpin(&d_matq_, d_random_, devStream_->getCudaStream());
    devCopy_.clearPadding(&d_matq_);
    setState(solQSet);
}

template<class real>
void CUDADenseGraphAnnealer<real>::calculate_E() {
    throwErrorIfQNotSet();

    DeviceVector *d_E = devStream_->tempDeviceVector<real>(m_);
    DeviceMatrix *d_realMatQ = devStream_->tempDeviceMatrix<real>(d_matq_.dim());
    devCopy_.cast(d_realMatQ, d_matq_);
    devFormulas_.calculate_E(d_E, d_h_, d_J_, d_c_, *d_realMatQ);
    real sign = (om_ == sq::optMaximize) ? real(-1.) : real(1.);
    devFormulas_.devMath.scale(&h_E_, sign, *d_E);
    /* FIXME: due to asynchronous execution, here is not a good place to set solEAvailable. */
    setState(solEAvailable);
}

template<class real>
void CUDADenseGraphAnnealer<real>::prepare() {
    throwErrorIfProblemNotSet();
    throwErrorIf(devStream_->getNumThreadsToFillDevice() < (m_ + 1) / 2,
                 "nTrotters too large for this device.");

    if (!isRandSeedGiven())
        d_random_.seed();
    setState(solRandSeedGiven);

    deallocateInternalObjects();

    if (m_ == 1) {
        /* force set to SANaive when m == 1. */
        if (algo_ != sq::algoSANaive) {
            algo_ = sq::algoSANaive;
            sq::log("algorithm set to sa_naive since m == 1.");
        }
    }
    
    switch (algo_) {
    case sq::algoColoring:
        annealMethod_ = &CUDADenseGraphAnnealer::annealOneStepSQA;
        break;
    case sq::algoSANaive:
        annealMethod_ = &CUDADenseGraphAnnealer::annealOneStepSA;
        break;
    default:
        abort_("Must not reach here.");
    }
    
    HostObjectAllocator halloc;
    devAlloc_->allocate(&d_matq_, m_, N_);
    devAlloc_->allocate(&d_Jq_, m_);
    halloc.allocate(&h_E_, m_);
    E_.map(h_E_.d_data, h_E_.size);
    halloc.allocate(&h_q_, sq::Dim(m_, N_));
    xlist_.reserve(m_);
    qlist_.reserve(m_);

    /* estimate # rand nums required per one anneal. */
    nRunsPerRandGen_ = maxRandBufCapacity / (m_ * N_ * sizeof(real));
    nRunsPerRandGen_ = std::max(2, std::min(nRunsPerRandGen_, (sq::SizeType)maxNRunsPerRandGen));
    sq::SizeType requiredSize = nRunsPerRandGen_ * m_ * N_ * sizeof(real) / sizeof(float);
    d_random_.setRequiredSize(requiredSize);

    throwOnError(cudaMemsetAsync(d_reachCount_, 0, sizeof(uint2), devStream_->getCudaStream()));

    DotJq<real> &dotJq = static_cast<DotJq<real>&>(*dotJq_);
    dotJq.configure(N_, m_, false);

    setState(solPrepared);
}

template<class real>
void CUDADenseGraphAnnealer<real>::makeSolution() {
    throwErrorIfQNotSet();
    syncBits();
    setState(solSolutionAvailable);
    calculate_E();
    // devStream_->synchronize();
}

template<class real>
void CUDADenseGraphAnnealer<real>::syncBits() {
    xlist_.clear();
    qlist_.clear();

    devCopy_(&h_q_, d_matq_);
    devStream_->synchronize();
    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        BitSet q(h_q_.row(idx), N_);
        qlist_.pushBack(q);
        BitSet x(qlist_.size());
        x = x_from_q(q);
        xlist_.pushBack(x);
    }
}

#if 0
/* equivalent code */
template<class real>
void annealOneStep(real G, real beta) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G * beta / m_)) * beta;

    for (int outer = 0; outer < IdxType(N_); ++outer) {
        int x[m];

        /* carried out in DeviceRandomBuffer. */
        for (int y = 0; y < IdxType(m_); ++y) {
            /* first plane */
            int fraction = y % 2;
            /* second plane */
            int fraction = 1 - y % 2;

            x[innder] = (random_random() * 2 + fraction) % N;
        }

        /* calculate_Jq() */
        real d_Jq[m];
        for (int y = 0; y < IdxType(m_); ++y)
            d_Jq[y] = J_.row(x[y]).dot(matQ_.row(y));

        /* flip each bit, */
        for (int inner = 0; inner < IdxType(m_); ++inner) {
            /* flip one bit */
            real qyx = matQ_(y, x[m]);

            real dE = twoDivM * qyx * (d_Jq[x[y] + h_(x[y])];
            int neibour0 = (m_ + y - 1) % m_, neibour1 = (y + 1) % m_;
            dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * beta);
            if (threshold > random_.random<real>())
                matQ_(y, x) = - qyx;
        }
    }
}
#endif

template<class real>
void CUDADenseGraphAnnealer<real>::calculate_Jq(DeviceVector *d_Jq,
                                                const DeviceMatrix &d_J, const DeviceBitMatrix &d_matq,
                                                const int *d_flipPos) {

    DotJq<real> &dotJq = static_cast<DotJq<real>&>(*dotJq_);
    dotJq(d_J, d_matq, d_flipPos, d_Jq->d_data);

}

template<bool mIsOdd, class real>
__global__ static void
tryFlipSQAKernel(char *d_q, sq::SizeType qStride, const real *d_Jq,
                 const real *d_h,
                 const int *d_x, const real *d_random, sq::SizeType m,
                 const real twoDivM, const real coef, const real beta,
                 uint2 *reachCount) {

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
#pragma unroll
    for (int offset = 0; offset < 2; ++offset) {
        if (gid < m / 2) {
            int y = 2 * gid + offset;
            int x = d_x[y]; /* N */
            char qyx = d_q[qStride * y + x];

            int neibour0 = (y == 0) ? m - 1 : y - 1;
            int neibour1 = (y == m - 1) ? 0 : y + 1;
            real dE = twoDivM * (real)qyx * (d_Jq[y] + d_h[x]);
            dE -= (real)qyx * (d_q[qStride * neibour0 + x] + d_q[qStride * neibour1 + x]) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : exp(-dE * beta);
            if (threshold > d_random[y])
                d_q[qStride * y + x] = - qyx;
        }
        if (offset == 0) {
            if ((gid == 0) && mIsOdd) {
                int y = m - 1;
                int x = d_x[y]; /* N */
                char qyx = d_q[qStride * y + x];

                int neibour0 = m - 2, neibour1 = 0;
                real dE = twoDivM * (real)qyx * (d_Jq[y] + d_h[x]);
                dE -= (real)qyx * (d_q[qStride * neibour0 + x] + d_q[qStride * neibour1 + x]) * coef;
                real threshold = (dE < real(0.)) ? real(1.) : exp(-dE * beta);
                if (threshold > d_random[y])
                    d_q[qStride * y + x] = - qyx;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int count = atomicAdd(&reachCount->x, 1) + 1;
                while (count != gridDim.x)
                    count = *(volatile unsigned int*)(&reachCount->x);
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int count = atomicAdd(&reachCount->y, 1) + 1;
        if (count == gridDim.x)
            *reachCount = make_uint2(0, 0);
    }
}

template<class real> void CUDADenseGraphAnnealer<real>::
annealOneStepSQA(DeviceBitMatrix *d_matq, const DeviceVector &d_Jq, const int *d_x,
                 const real *d_random,
                 const DeviceVector &d_h, const DeviceMatrix &d_J, real G, real beta) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G * beta / m_)) * beta;

    dim3 blockDim(128);

    int nThreadsToFlipBits = m_ / 2;
    dim3 gridDim(std::max(divru(nThreadsToFlipBits, blockDim.x), 1));
    gridDim.x = std::min((unsigned int)1, gridDim.x); /* for the case of m == 1. */
    cudaStream_t stream = devStream_->getCudaStream();
    bool mIsOdd = (m_ & 1) != 0;
#if 0
    if (mIsOdd) {
        tryFlipKernelSQA<true><<<gridDim, blockDim, 0, stream>>>(d_matq->d_data, d_matq->stride,
                                                                 d_Jq.d_data, d_h.d_data,
                                                                 d_x, d_random, m_,
                                                                 twoDivM, coef, beta, d_reachCount_);
    }
    else {
        tryFlipKernelSQA<false><<<gridDim, blockDim, 0, stream>>>(d_matq->d_data, d_matq->stride,
                                                                  d_Jq.d_data, d_h.d_data,
                                                                  d_x, d_random, m_,
                                                                  twoDivM, coef, beta, d_reachCount_);
    }
#else
    void *args[] = {(void*)&d_matq->d_data, (void*)&d_matq->stride,
                    (void*)&d_Jq.d_data, (void*)&d_h.d_data, (void*)&d_x, (void*)&d_random, 
                    (void*)&m_, (void*)&twoDivM, (void*)&coef, (void*)&beta, (void*)&d_reachCount_, NULL};
    if (mIsOdd)
        cudaLaunchKernel((void*)tryFlipSQAKernel<true, real>, gridDim, blockDim, args, 0, stream);
    else
        cudaLaunchKernel((void*)tryFlipSQAKernel<false, real>, gridDim, blockDim, args, 0, stream);
#endif
    DEBUG_SYNC;
}



template<class real>
void CUDADenseGraphAnnealer<real>::annealOneStepSQA(real G, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);
    
    if (!flipPosBuffer_.available(m_ * N_))
        flipPosBuffer_.generateFlipPositions(d_random_, N_, m_, nRunsPerRandGen_);
    if (!realNumBuffer_.available(m_ * N_))
        realNumBuffer_.generate<real>(d_random_, N_ * m_ * nRunsPerRandGen_);
    for (int idx = 0; idx < N_; ++idx) {
        const int *d_flipPos = flipPosBuffer_.acquire<int>(m_);
        const real *d_random = realNumBuffer_.acquire<real>(m_);
        calculate_Jq(&d_Jq_, d_J_, d_matq_, d_flipPos);
        annealOneStepSQA(&d_matq_, d_Jq_, d_flipPos, d_random, d_h_, d_J_, G, beta);
    }
}


/* Simulated annealing */

template<class real>
__global__ static void
tryFlipSAKernel(char *d_q, sq::SizeType qStride, const real *d_Jq,
                const real *d_h,
                const int *d_x, const real *d_random, sq::SizeType m,
                const real Tnorm) {

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < m) {
        int y = gid;
        int x = d_x[y]; /* N */
        char qyx = d_q[qStride * y + x];
        
        real dE = real(2.) * (real)qyx * (d_Jq[y] + d_h[x]);
        real threshold = (dE < real(0.)) ? real(1.) : exp(-dE * Tnorm);
        if (threshold > d_random[y])
            d_q[qStride * y + x] = - qyx;
    }
}

template<class real> void CUDADenseGraphAnnealer<real>::
annealOneStepSA(DeviceBitMatrix *d_matq, const DeviceVector &d_Jq, const int *d_x,
                const real *d_random,
                const DeviceVector &d_h, const DeviceMatrix &d_J, real Tnorm) {
    dim3 blockDim(128);
    dim3 gridDim(std::max(divru(m_, blockDim.x), 1));
    cudaStream_t stream = devStream_->getCudaStream();
#if 0
    tryFlipSAKernel<real><<<gridDim, blockDim, 0, stream>>>(d_matq->d_data, d_matq->stride,
                                                            d_Jq.d_data, d_h.d_data,
                                                            d_x, d_random, m_,
                                                            Tnorm);
#else
    void *args[] = {(void*)&d_matq->d_data, (void*)&d_matq->stride,
                    (void*)&d_Jq.d_data, (void*)&d_h.d_data, (void*)&d_x, (void*)&d_random, 
                    (void*)&m_, (void*)&Tnorm, NULL};
    cudaLaunchKernel((void*)tryFlipSAKernel<real>, gridDim, blockDim, args, 0, stream);
#endif
    DEBUG_SYNC;
}


template<class real>
void CUDADenseGraphAnnealer<real>::annealOneStepSA(real kT, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);
    
    if (!flipPosBuffer_.available(m_ * N_))
        flipPosBuffer_.generateFlipPositions(d_random_, N_, m_, nRunsPerRandGen_);
    if (!realNumBuffer_.available(m_ * N_))
        realNumBuffer_.generate<real>(d_random_, N_ * m_ * nRunsPerRandGen_);

    real Tnorm = kT * beta;
    for (int idx = 0; idx < N_; ++idx) {
        const int *d_flipPos = flipPosBuffer_.acquire<int>(m_);
        const real *d_random = realNumBuffer_.acquire<real>(m_);
        calculate_Jq(&d_Jq_, d_J_, d_matq_, d_flipPos);
        annealOneStepSA(&d_matq_, d_Jq_, d_flipPos, d_random, d_h_, d_J_, Tnorm);
    }
}




template class CUDADenseGraphAnnealer<double>;
template class CUDADenseGraphAnnealer<float>;
