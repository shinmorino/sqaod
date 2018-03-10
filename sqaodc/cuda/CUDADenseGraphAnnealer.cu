#include "CUDADenseGraphAnnealer.h"
#include "DeviceKernels.h"
#include "cub_iterator.cuh"
#include <cub/cub.cuh>
#include "DeviceSegmentedSum.cuh"

using namespace sqaod_cuda;

template<class real>
CUDADenseGraphAnnealer<real>::CUDADenseGraphAnnealer() {
    devStream_ = NULL;
    m_ = (SizeType)-1;
    d_reachCount_ = NULL;
}

template<class real>
CUDADenseGraphAnnealer<real>::CUDADenseGraphAnnealer(Device &device) {
    devStream_ = NULL;
    m_ = (SizeType)-1;
    d_reachCount_ = NULL;
    assignDevice(device);
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
void CUDADenseGraphAnnealer<real>::assignDevice(Device &device) {
    throwErrorIf(devStream_ != NULL, "Device assigned more than once.");
    devStream_ = device.defaultStream();
    devAlloc_ = device.objectAllocator();
    devFormulas_.assignDevice(device);
    devCopy_.assignDevice(device);
    d_random_.assignDevice(device);
    flipPosBuffer_.assignDevice(device);
    realNumBuffer_.assignDevice(device);

    d_reachCount_ = (uint2*)devAlloc_->allocate(sizeof(uint2));

    /* initialize sumJq */
    typedef DeviceSegmentedSumTypeImpl<real, In2TypeDotPtr<real, char, real>, real*, Offset2way> DotJq;
    dotJq_ = new DotJq(device);
}

template<class real>
sq::Algorithm CUDADenseGraphAnnealer<real>::selectAlgorithm(sq::Algorithm algo) {
    return sq::algoColoring;
}

template<class real>
sq::Algorithm CUDADenseGraphAnnealer<real>::getAlgorithm() const {
    return sq::algoColoring;
}


template<class real>
void CUDADenseGraphAnnealer<real>::seed(unsigned long long seed) {
    throwErrorIf(devStream_ == NULL, "Device not set.");
    d_random_.seed(seed);
    setState(solRandSeedGiven);
}

template<class real>
void CUDADenseGraphAnnealer<real>::setProblem(const HostMatrix &W, sq::OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    throwErrorIf(devStream_ == NULL, "Device not set.");
    deallocate();
    clearState(solProblemSet);

    N_ = W.rows;
    m_ = N_ / 4;
    om_ = om;

    DeviceMatrix *dW = devStream_->tempDeviceMatrix<real>(W.dim(), __func__);
    devCopy_(dW, W);
    if (om == sq::optMaximize)
        devFormulas_.devMath.scale(dW, -1., *dW);
    devFormulas_.calculate_hJc(&d_h_, &d_J_, &d_c_, *dW);

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
    if (!isEAvailable()) {
        const_cast<This*>(this)->calculate_E();
        devStream_->synchronize();
    }
    return E_;
}

template<class real>
const sq::BitsArray &CUDADenseGraphAnnealer<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return xlist_;
}


template<class real>
void CUDADenseGraphAnnealer<real>::set_x(const Bits &x) {
    throwErrorIfNotPrepared();
    throwErrorIf(x.size != N_,
                 "Dimension of x, %d,  should be equal to N, %d.", x.size, N_);

    DeviceBits *d_x = devStream_->tempDeviceVector<char>(x.size);
    devCopy_(d_x, x);
    devCopy_.copyRowwise(&d_matq_, *d_x);
    setState(solQSet);
}


template<class real>
void CUDADenseGraphAnnealer<real>::get_hJc(HostVector *h, HostMatrix *J, real *c) const {
    throwErrorIfProblemNotSet();

    devCopy_(h, d_h_);
    devCopy_(J, d_J_);
    devCopy_(c, d_c_);
    devCopy_.synchronize();
}

template<class real>
const sq::BitsArray &CUDADenseGraphAnnealer<real>::get_q() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return qlist_;
}

template<class real>
void CUDADenseGraphAnnealer<real>::randomize_q() {
    throwErrorIfNotPrepared();

    ::randomize_q(d_matq_.d_data, d_random_, d_matq_.rows * d_matq_.cols,
                  devStream_->getCudaStream());
    setState(solQSet);
}

template<class real>
void CUDADenseGraphAnnealer<real>::calculate_E() {
    throwErrorIfQNotSet();

    DeviceVector *d_E = devStream_->tempDeviceVector<real>(m_);
    DeviceMatrix *d_realMatQ = devStream_->tempDeviceMatrix<real>(d_matq_.dim());
    devCopy_.cast(d_realMatQ, d_matq_);
    devFormulas_.calculate_E(d_E, d_h_, d_J_, d_c_, *d_realMatQ);
    real sign = (om_ == sq::optMaximize) ? -1. : 1.;
    devFormulas_.devMath.scale(&h_E_, sign, *d_E);

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

    HostObjectAllocator halloc;
    devAlloc_->allocate(&d_matq_, m_, N_);
    devAlloc_->allocate(&d_Jq_, m_);
    halloc.allocate(&h_E_, m_);
    E_.map(h_E_.d_data, h_E_.size);
    halloc.allocate(&h_q_, sq::Dim(m_, N_));
    xlist_.reserve(m_);
    qlist_.reserve(m_);
    /* estimate # rand nums required per one anneal. */
    int requiredSize = (N_ * m_ * (nRunsPerRandGen + 1)) * sizeof(real) / 4;
    d_random_.setRequiredSize(requiredSize);
    throwOnError(cudaMemsetAsync(d_reachCount_, 0, sizeof(uint2), devStream_->getCudaStream()));

    typedef DeviceSegmentedSumTypeImpl<real, In2TypeDotPtr<real, char, real>, real*, Offset2way> DotJq;
    DotJq &dotJq = static_cast<DotJq&>(*dotJq_);
    dotJq.configure(N_, m_, false);

    setState(solPrepared);
}

template<class real>
void CUDADenseGraphAnnealer<real>::makeSolution() {
    throwErrorIfQNotSet();
    syncBits();
    setState(solSolutionAvailable);
    calculate_E();
}

template<class real>
void CUDADenseGraphAnnealer<real>::syncBits() {
    xlist_.clear();
    qlist_.clear();

    devCopy_(&h_q_, d_matq_);
    devStream_->synchronize();
    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        Bits q(h_q_.row(idx), N_);
        qlist_.pushBack(q);
        Bits x(sq::SizeType(qlist_.size()));
        x = x_from_q(q);
        xlist_.pushBack(x);
    }
}

#if 0
/* equivalent code */
template<class real>
void annealOneStep(real G, real kT) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;

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

            real dE = - twoDivM * qyx * (d_Jq[x[y] + h_(x[y])];
            int neibour0 = (m_ + y - 1) % m_, neibour1 = (y + 1) % m_;
            dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE / kT);
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
    cudaStream_t stream = devStream_->getCudaStream();
    In2TypeDotPtr<real, char, real> inPtr(d_matq.d_data, d_J.d_data);
    typedef DeviceSegmentedSumTypeImpl<real, In2TypeDotPtr<real, char, real>, real*, Offset2way> DotJq;
    DotJq &dotJq = static_cast<DotJq&>(*dotJq_);
    dotJq(inPtr, d_Jq->d_data, Offset2way(d_flipPos, N_));
}

template<class real>
__global__ static void
tryFlipKernel(char *d_q, const real *d_Jq, const real *d_h,
              const int *d_x, const real *d_random, sq::SizeType N, sq::SizeType m,
              const real twoDivM, const real coef, const real invKT,
              uint2 *reachCount) {

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int y = 2 * gid;
    for (int loop = 0; loop < 2; ++loop) {
        if (y < m) {
            int x = d_x[y]; /* N */
            char qyx = d_q[N * y + x];

            int neibour0 = (y == 0) ? m - 1 : y - 1;
            int neibour1 = (y == m - 1) ? 0 : y + 1;
            real dE = - twoDivM * (real)qyx * (d_Jq[y] + d_h[x]);
            dE -= (real)qyx * (d_q[N * neibour0 + x] + d_q[N * neibour1 + x]) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : exp(- dE * invKT);
            if (threshold > d_random[y])
                d_q[N * y + x] = - qyx;
        }

        /* wait for all blocks reach here */
        __syncthreads();
        if ((loop == 0) && (threadIdx.x == 0)) {
            int count = atomicAdd(&reachCount->x, 1) + 1;
            while (count != gridDim.x) {
                count = *(volatile unsigned int*)(&reachCount->x);
            }
        }
        __syncthreads();
        y += 1;
    }
    if (threadIdx.x == 0) {
        int count = atomicAdd(&reachCount->y, 1) + 1;
        if (count == gridDim.x)
            *reachCount = make_uint2(0, 0);
    }
}

template<class real> void CUDADenseGraphAnnealer<real>::
annealOneStep(DeviceBitMatrix *d_matq, const DeviceVector &d_Jq, const int *d_x, const real *d_random,
              const DeviceVector &d_h, const DeviceMatrix &d_J, real G, real kT) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
    real invKT = real(1.) / kT;

    dim3 blockDim(128);

    int nThreadsToFlipBits = (m_ + 1) / 2;
    dim3 gridDim(divru((sq::SizeType)nThreadsToFlipBits, blockDim.x));
    cudaStream_t stream = devStream_->getCudaStream();
#if 0
    tryFlipKernel<<<gridDim, blockDim, 0, stream>>>(d_matq->d_data, d_Jq.d_data, d_h.d_data,
                                         d_x, d_random, N_, m_,
                                         twoDivM, coef, invKT, d_reachCount_);
#else
    void *args[] = {(void*)&d_matq->d_data, (void*)&d_Jq.d_data, (void*)&d_h.d_data, (void*)&d_x, (void*)&d_random, 
                    (void*)&N_, (void*)&m_, (void*)&twoDivM, (void*)&coef, (void*)&invKT, (void*)&d_reachCount_, NULL};
    cudaLaunchKernel((void*)tryFlipKernel<real>, gridDim, blockDim, args, 0, stream);
#endif
    DEBUG_SYNC;
}



template<class real>
void CUDADenseGraphAnnealer<real>::annealOneStep(real G, real kT) {
    throwErrorIfQNotSet();
    clearState(solEAvailable);
    clearState(solSolutionAvailable);
    
    if (!flipPosBuffer_.available(m_ * N_))
        flipPosBuffer_.generateFlipPositions(d_random_, N_, m_, nRunsPerRandGen);
    if (!realNumBuffer_.available(m_ * N_))
        realNumBuffer_.generate<real>(d_random_, N_ * m_ * nRunsPerRandGen);
    for (int idx = 0; idx < N_; ++idx) {
        const int *d_flipPos = flipPosBuffer_.acquire<int>(m_);
        const real *d_random = realNumBuffer_.acquire<real>(m_);
        calculate_Jq(&d_Jq_, d_J_, d_matq_, d_flipPos);
        annealOneStep(&d_matq_, d_Jq_, d_flipPos, d_random, d_h_, d_J_, G, kT);
    }
    clearState(solEAvailable);
}




template class CUDADenseGraphAnnealer<double>;
template class CUDADenseGraphAnnealer<float>;
