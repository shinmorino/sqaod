#include "CUDABipartiteGraphAnnealer.h"
#include "devfuncs.cuh"
#include <sqaodc/common/internal/ShapeChecker.h>
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>

namespace sqint = sqaod_internal;
using namespace sqaod_cuda;

template<class real>
CUDABipartiteGraphAnnealer<real>::CUDABipartiteGraphAnnealer() {
    devStream_ = NULL;
    m_ = -1;
    selectAlgorithm(sq::algoDefault);
}

template<class real>
CUDABipartiteGraphAnnealer<real>::CUDABipartiteGraphAnnealer(Device &device) {
    devStream_ = NULL;
    m_ = -1;
    assignDevice(device);
    selectAlgorithm(sq::algoDefault);
}

template<class real>
CUDABipartiteGraphAnnealer<real>::~CUDABipartiteGraphAnnealer() {
    deallocate();
    d_random_.deallocate();
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::deallocateProblem() {
    devAlloc_->deallocate(d_h0_);
    devAlloc_->deallocate(d_h1_);
    devAlloc_->deallocate(d_J_);
    devAlloc_->deallocate(d_c_);

    clearState(solProblemSet);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::deallocateInternalObjects() {
    devAlloc_->deallocate(d_matq0_);
    devAlloc_->deallocate(d_matq1_);
    devAlloc_->deallocate(d_Jq0_);
    devAlloc_->deallocate(d_Jq1_);
    
    HostObjectAllocator halloc;
    halloc.deallocate(h_q0_);
    halloc.deallocate(h_q1_);
    halloc.deallocate(h_E_);
    E_ = HostVector();
    
    d_randReal_.deallocate();
    
    clearState(solPrepared);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::deallocate() {
    deallocateProblem();
    deallocateInternalObjects();
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::assignDevice(sqaod::cuda::Device &device) {
    assignDevice(static_cast<Device&>(device));
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::assignDevice(Device &device) {
    devStream_ = device.defaultStream();
    devAlloc_ = device.objectAllocator();
    devFormulas_.assignDevice(device);
    devCopy_.assignDevice(device);
    d_random_.assignDevice(device);
    d_randReal_.assignDevice(device);
}

template<class real>
sq::Algorithm CUDABipartiteGraphAnnealer<real>::selectAlgorithm(sq::Algorithm algo) {
    switch (algo) {
    case sq::algoColoring:
    case sq::algoSAColoring:
        algo_ = algo;
        break;
    default:
        selectDefaultAlgorithm(algo, sq::algoColoring, sq::algoSAColoring);
        break;
    }
    return algo_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::seed(unsigned long long seed) {
    throwErrorIf(devStream_ == NULL, "Device not set.");
    d_random_.seed(seed);
    setState(solRandSeedGiven);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::
setQUBO(const HostVector &b0, const HostVector &b1, const HostMatrix &W, sq::OptimizeMethod om) {
    sqint::quboShapeCheck(b0, b1, W, __func__);
    throwErrorIf(devStream_ == NULL, "Device not set.");
    if ((W.cols != N0_) || (W.rows != N1_))
        deallocate();

    N0_ = W.cols;
    N1_ = W.rows;
    m_ = (N0_ + N1_) / 4;
    om_ = om;

    DeviceVector *d_b0 = devStream_->tempDeviceVector<real>(b0.size);
    DeviceVector *d_b1 = devStream_->tempDeviceVector<real>(b1.size);
    DeviceMatrix *d_W = devStream_->tempDeviceMatrix<real>(W.dim());
    devCopy_(d_b0, b0);
    devCopy_(d_b1, b1);
    devCopy_(d_W, W);
    if (om == sq::optMaximize) {
        devFormulas_.devMath.scale(d_b0, -1., *d_b0);
        devFormulas_.devMath.scale(d_b1, -1., *d_b1);
        devFormulas_.devMath.scale(d_W, -1., *d_W);
    }

    devFormulas_.calculateHamiltonian(&d_h0_, &d_h1_, &d_J_, &d_c_, *d_b0, *d_b1, *d_W);
    devStream_->synchronize();

    setState(solProblemSet);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::
setHamiltonian(const HostVector &h0, const HostVector &h1, const HostMatrix &J, real c) {
    throwErrorIf(devStream_ == NULL, "Device not set.");
    sqint::isingModelShapeCheck(h0, h1, J, c, __func__);
    deallocate();

    N0_ = J.cols;
    N1_ = J.rows;
    m_ = (N0_ + N1_) / 4;
    om_ = sq::optMinimize;

    devCopy_(&d_h0_, h0);
    devCopy_(&d_h1_, h1);
    devCopy_(&d_J_, J);
    devCopy_(&d_c_, c);
    devStream_->synchronize();

    setState(solProblemSet);
}

template<class real>
sq::Preferences CUDABipartiteGraphAnnealer<real>::getPreferences() const {
    sq::Preferences prefs = Base::getPreferences();
    prefs.pushBack(sq::Preference(sq::pnDevice, "cuda"));
    return prefs;
}

template<class real>
const sq::VectorType<real> &CUDABipartiteGraphAnnealer<real>::get_E() const {
    if (!isEAvailable())
        const_cast<This*>(this)->calculate_E();
    /*  FIXME: Add a flag to show kernel is not synchronized */
    devStream_->synchronize();
    return E_;
}

template<class real>
const sq::BitSetPairArray &CUDABipartiteGraphAnnealer<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return bitsPairX_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::set_q(const sq::BitSetPair &qPair) {
    sqint::isingModelSolutionShapeCheck(N0_, N1_, qPair.bits0, qPair.bits1, __func__);
    throwErrorIfNotPrepared();

    HostVector rx0 = sq::cast<real>(qPair.bits0);
    HostVector rx1 = sq::cast<real>(qPair.bits1);
    DeviceVector *d_x0 = devStream_->tempDeviceVector<real>(rx0.size);
    DeviceVector *d_x1 = devStream_->tempDeviceVector<real>(rx1.size);
    devCopy_(d_x0, rx0);
    devCopy_(d_x1, rx1);
    devCopy_.synchronize(); /* rx0, rx1 are on stack. */
    devFormulas_.devMath.scaleBroadcast(&d_matq0_, real(1.), *d_x0, opRowwise);
    devFormulas_.devMath.scaleBroadcast(&d_matq1_, real(1.), *d_x1, opRowwise);
    devStream_->synchronize();

    setState(solQSet);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::set_qset(const sq::BitSetPairArray &qPairs) {
    sqint::isingModelSolutionShapeCheck(N0_, N1_, qPairs, __func__);
    m_ = qPairs.size();
    prepare();

    HostMatrix matq0(m_, N0_), matq1(m_, N1_);
    for (int idx = 0; idx < m_; ++idx) {
        HostVector rx0 = sq::cast<real>(qPairs[idx].bits0);
        HostVector rx1 = sq::cast<real>(qPairs[idx].bits1);
        memcpy(&matq0(idx, 0), rx0.data, sizeof(real) * N0_);
        memcpy(&matq1(idx, 0), rx1.data, sizeof(real) * N1_);
    }
    
    devCopy_(&d_matq0_, matq0);
    devCopy_(&d_matq1_, matq1);
    devCopy_.synchronize(); /* rx0, rx1 are on stack. */
    setState(solQSet);
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::getHamiltonian(HostVector *h0, HostVector *h1,
                                                      HostMatrix *J, real *c) const {
    throwErrorIfProblemNotSet();

    devCopy_(h0, d_h0_);
    devCopy_(h1, d_h1_);
    devCopy_(J, d_J_);
    devCopy_(c, d_c_);
    devCopy_.synchronize();
}


template<class real>
const sq::BitSetPairArray &CUDABipartiteGraphAnnealer<real>::get_q() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return bitsPairQ_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::randomizeSpin() {
    throwErrorIfNotPrepared();

    cudaStream_t stream = devStream_->getCudaStream();
    sqaod_cuda::randomizeSpin(&d_matq0_, d_random_, stream);
    sqaod_cuda::randomizeSpin(&d_matq1_, d_random_, stream);
    setState(solQSet);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::calculate_E() {
    throwErrorIfQNotSet();

    DeviceVector *d_E = devStream_->tempDeviceVector<real>(m_);
    devFormulas_.calculate_E(d_E, d_h0_, d_h1_, d_J_, d_c_,
                             d_matq0_, d_matq1_);
    real sign = (om_ == sq::optMaximize) ? real(-1.) : real(1.);
    devFormulas_.devMath.scale(&h_E_, sign, *d_E);

    setState(solEAvailable);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::prepare() {
    throwErrorIfProblemNotSet();

    if (!isRandSeedGiven())
        d_random_.seed();
    setState(solRandSeedGiven);

    deallocateInternalObjects();

    if (m_ == 1)
        selectDefaultSAAlgorithm(algo_, sq::algoSAColoring);

    switch (algo_) {
    case sq::algoColoring:
        annealMethod_ = &CUDABipartiteGraphAnnealer::annealOneStepSQA;
        break;
    case sq::algoSAColoring:
        annealMethod_ = &CUDABipartiteGraphAnnealer<real>::annealOneStepSA;
        break;
    default:
        abort_("Must not reach here.");
    }

    devAlloc_->allocate(&d_matq0_, m_, N0_);
    devAlloc_->allocate(&d_matq1_, m_, N1_);

    HostObjectAllocator halloc;
    halloc.allocate(&h_E_, m_);
    E_.map(h_E_.d_data, h_E_.size);
    halloc.allocate(&h_q0_, m_, N0_);
    halloc.allocate(&h_q1_, m_, N1_);
    bitsPairX_.reserve(m_);
    bitsPairQ_.reserve(m_);

    /* estimate # rand nums required per one anneal. */
    sq::SizeType N = N0_ + N1_;
    nRunsPerRandGen_ = maxRandBufCapacity / (m_ * N * sizeof(real));
    nRunsPerRandGen_ = std::max(2, std::min(nRunsPerRandGen_, (sq::SizeType)maxNRunsPerRandGen));
    sq::SizeType requiredSize = nRunsPerRandGen_ * m_ * N * sizeof(real) / sizeof(float);
    d_random_.setRequiredSize(requiredSize);

    setState(solPrepared);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::makeSolution() {
    throwErrorIfQNotSet();

    syncBits();
    calculate_E();
    devStream_->synchronize();

    setState(solSolutionAvailable);
}



// template<class real>
// void CUDABipartiteGraphAnnealer<real>::
// annealHalfStep(DeviceMatrix *d_qAnneal, int N,
//                const DeviceVector &d_h, real G, real beta) {
//     real twoDivM = real(2.) / m_;
//     bgFuncs_.devMath.matmul(d_Jq, d_J, op, qFixed, opTranspose);
//     real tempCoef = std::log(std::tanh(G * beta / m_)) * beta;
//     for (int loop = 0; loop < IdxType(N * m_); ++loop) {
//         real q = qAnneal(im, iq);
//         real dE = twoDivM * q * (h[iq] + dEmat(iq, im));
//         int mNeibour0 = (im + m_ - 1) % m_;
//         int mNeibour1 = (im + 1) % m_;
//         dE -= q * (qAnneal(mNeibour0, iq) + qAnneal(mNeibour1, iq)) * tempCoef;
//         real thresh = dE < real(0.) ? real(1.) : std::exp(-dE * beta);
//         if (thresh > random_.random<real>())
//             qAnneal(im, iq) = -q;
//     }
// }                    


template<class real> void CUDABipartiteGraphAnnealer<real>::
calculate_Jq(DeviceMatrix *d_Jq, const DeviceMatrix &d_J, MatrixOp op,
             const DeviceMatrix &d_qFixed) {
    /* original matmul is shown below.
     *  bgFuncs_.devMath.mmProduct(d_Jq, 1., d_J, op, d_qFixed, opTranspose);
     * Tranpose product for coalesced access. */

    op = (op == opNone) ? opTranspose : opNone;
    devFormulas_.devMath.mmProduct(d_Jq, 1., d_qFixed, opNone, d_J, op);
}

#if 0

template<int offset, class real>
__global__ static void
tryFlipKernel(real *d_qAnneal, sq::SizeType qAnnealStride,
              int N, int m, const real *d_Emat, sq::SizeType EmatStride,
              const real *d_h, const real *d_realRand,
              real twoDivM, real coef, real beta, bool runLastLine) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int iq = gidx;
    int m2 = m / 2; /* round down */

    if ((iq < N) && (gidy < m2)) {
        int im = 2 * gidy + offset;
        real q = d_qAnneal[im * qAnnealStride + iq];
        real dE = twoDivM * q * (d_h[iq] + d_Emat[im * qAnnealStride + iq]);

        int neibour0 = (im == 0) ? m - 1 : im - 1;
        int neibour1 = (im == m - 1) ? 0 : im + 1;
        dE -= q * (d_qAnneal[neibour0 * qAnnealStride + iq] + d_qAnneal[neibour1 * qAnnealStride + iq]) * coef;
        real thresh = dE < real(0.) ? real(1.) : exp(-dE * beta);
        if (thresh > d_realRand[N * gidy + iq])
            d_qAnneal[im * qAnnealStride + iq] = - q;
    }
    if ((offset == 0) && runLastLine && (gidy == 0)) {
        int im = m - 1;
        if (iq < N) {
            real q = d_qAnneal[im * qAnnealStride + iq];
            real dE = twoDivM * q * (d_h[iq] + d_Emat[im * EmatStride + iq]);

            int neibour0 = im - 2;
            int neibour1 = 0;
            dE -= q * (d_qAnneal[neibour0 * qAnnealStride + iq] + d_qAnneal[neibour1 * qAnnealStride + iq]) * coef;
            real thresh = dE < real(0.) ? real(1.) : exp(-dE * beta);
            if (thresh > d_realRand[N * gidy + iq])
                d_qAnneal[im * qAnnealStride + iq] = - q;
        }
    }
}

template<class real> void CUDABipartiteGraphAnnealer<real>::
tryFlip(DeviceMatrix *d_qAnneal, const DeviceMatrix &d_Jq, int N, int m,
        const DeviceVector &d_h, const real *d_realRand, real G, real beta) {
    real coef = std::log(std::tanh(G * beta / m_)) * beta;
    real twoDivM = real(2.) / m_;
    int m2 = m_ / 2;
    bool mIsOdd = (m_ & 1) != 0;

    dim3 blockDim(64, 2);
    dim3 gridDim(divru(N, blockDim.x), divru(m2, blockDim.y));
    tryFlipKernel<0><<<gridDim, blockDim>>>
            (d_qAnneal->d_data, d_qAnneal->stride,
             N, m_, d_Jq.d_data, d_Jq.stride, d_h.d_data, d_realRand,
             twoDivM, coef, beta, mIsOdd);
    DEBUG_SYNC;
    tryFlipKernel<1><<<gridDim, blockDim>>>
            (d_qAnneal->d_data, d_qAnneal->stride,
             N, m_, d_Jq.d_data, d_Jq.stride, d_h.d_data, d_realRand,
             twoDivM, coef, beta, false);
    DEBUG_SYNC;
}

#else

template<int offset, class real>
__device__ __forceinline__ static void
deviceTryFlipSQA(int gidx, int gidy, real *d_qAnneal, sq::SizeType qAnnealStride,
                 int N, int m, const real *d_Emat, sq::SizeType EmatStride,
                 const real *d_h, const real *d_realRand,
                 real twoDivM, real coef, real beta, bool runLastLine) {
    int iq = gidx;
    int m2 = m / 2; /* round down */

    if ((iq < N) && (gidy < m2)) {
        int im = 2 * gidy + offset;
        real q = d_qAnneal[im * qAnnealStride + iq];
        real dE = twoDivM * q * (d_h[iq] + d_Emat[im * qAnnealStride + iq]);

        int neibour0 = (im == 0) ? m - 1 : im - 1;
        int neibour1 = (im == m - 1) ? 0 : im + 1;
        dE -= q * (d_qAnneal[neibour0 * qAnnealStride + iq] + d_qAnneal[neibour1 * qAnnealStride + iq]) * coef;
        real thresh = dE < real(0.) ? real(1.) : exp(-dE * beta);
        if (thresh > d_realRand[N * gidy + iq])
            d_qAnneal[im * qAnnealStride + iq] = - q;
    }
    if ((offset == 0) && runLastLine && (gidy == 0)) {
        int im = m - 1;
        if (iq < N) {
            real q = d_qAnneal[im * qAnnealStride + iq];
            real dE = twoDivM * q * (d_h[iq] + d_Emat[im * EmatStride + iq]);

            int neibour0 = im - 2;
            int neibour1 = 0;
            dE -= q * (d_qAnneal[neibour0 * qAnnealStride + iq] + d_qAnneal[neibour1 * qAnnealStride + iq]) * coef;
            real thresh = dE < real(0.) ? real(1.) : exp(-dE * beta);
            if (thresh > d_realRand[N * gidy + iq])
                d_qAnneal[im * qAnnealStride + iq] = - q;
        }
    }
}

template<class real> void CUDABipartiteGraphAnnealer<real>::
tryFlipSQA(DeviceMatrix *d_qAnneal, const DeviceMatrix &d_Jq, int N, int m,
           const DeviceVector &d_h, const real *d_realRand, real G, real beta) {
    real coef = std::log(std::tanh(G * beta / m_)) * beta;
    real twoDivM = real(2.) / m_;
    int m2 = m_ / 2;
    bool mIsOdd = (m_ & 1) != 0;

    real *d_qAnneal_data = d_qAnneal->d_data;
    sq::SizeType qAnnealStride = d_qAnneal->stride;
    const real *d_Emat = d_Jq.d_data;
    sq::SizeType EmatStride = d_Jq.stride;
    const real *d_h_data = d_h.d_data;
    
    cudaStream_t stream = devStream_->getCudaStream();

    auto flipOp0 = [=]__device__(int gidx, int gidy) {
        deviceTryFlipSQA<0>(gidx, gidy,
                            d_qAnneal_data, qAnnealStride, N, m, d_Emat, EmatStride,
                            d_h_data, d_realRand, twoDivM, coef, beta, mIsOdd);
    };
    transform2d(flipOp0, N, m2, dim3(64, 2), stream);

    auto flipOp1 = [=]__device__(int gidx, int gidy) {
        deviceTryFlipSQA<1>(gidx, gidy,
                            d_qAnneal_data, qAnnealStride, N, m, d_Emat, EmatStride,
                            d_h_data, d_realRand, twoDivM, coef, beta, false);
    };
    transform2d(flipOp1, N, m2, dim3(64, 2), stream);
}

#endif

template<class real>
void CUDABipartiteGraphAnnealer<real>::annealOneStepSQA(real G, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);

    int nRequiredRandNum = (N0_ + N1_) * m_;
    if (!d_randReal_.available(nRequiredRandNum))
        d_randReal_.generate<real>(d_random_, nRequiredRandNum);
    const real *d_randNum;

    /* FIXME: consider Jq to use half trotters. */
    /* 1st */
    calculate_Jq(&d_Jq1_, d_J_, opNone, d_matq0_);
    d_randNum = d_randReal_.acquire<real>(N1_ * m_);
    tryFlipSQA(&d_matq1_, d_Jq1_, N1_, m_, d_h1_, d_randNum, G, beta);
    DEBUG_SYNC;

    /* 2nd */
    calculate_Jq(&d_Jq0_, d_J_, opTranspose, d_matq1_);
    d_randNum = d_randReal_.acquire<real>(N0_ * m_);
    tryFlipSQA(&d_matq0_, d_Jq0_, N0_, m_, d_h0_, d_randNum, G, beta);
    DEBUG_SYNC;
}


template<class real>
__device__ __forceinline__ static void
deviceTryFlipSA(int gidx, int gidy, real *d_qAnneal, sq::SizeType qAnnealStride,
                int N, int m, const real *d_Emat, sq::SizeType EmatStride,
                const real *d_h, const real *d_realRand,
                real Tnorm) {
    int iq = gidx;

    if ((iq < N) && (gidy < m)) {
        int im = gidy;
        real q = d_qAnneal[im * qAnnealStride + iq];
        real dE = real(2.) * q * (d_h[iq] + d_Emat[im * qAnnealStride + iq]);
        real thresh = dE < real(0.) ? real(1.) : exp(-dE * Tnorm);
        if (thresh > d_realRand[N * gidy + iq])
            d_qAnneal[im * qAnnealStride + iq] = - q;
    }
}

template<class real> void CUDABipartiteGraphAnnealer<real>::
tryFlipSA(DeviceMatrix *d_qAnneal, const DeviceMatrix &d_Jq, int N, int m,
          const DeviceVector &d_h, const real *d_realRand, real Tnorm) {

    real *d_qAnneal_data = d_qAnneal->d_data;
    sq::SizeType qAnnealStride = d_qAnneal->stride;
    const real *d_Emat = d_Jq.d_data;
    sq::SizeType EmatStride = d_Jq.stride;
    const real *d_h_data = d_h.d_data;
    
    cudaStream_t stream = devStream_->getCudaStream();

    auto flipOp0 = [=]__device__(int gidx, int gidy) {
        deviceTryFlipSA(gidx, gidy,
                        d_qAnneal_data, qAnnealStride, N, m, d_Emat, EmatStride,
                        d_h_data, d_realRand, Tnorm);
    };
    transform2d(flipOp0, N, m, dim3(64, 2), stream);
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::annealOneStepSA(real kT, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);

    int nRequiredRandNum = (N0_ + N1_) * m_;
    if (!d_randReal_.available(nRequiredRandNum))
        d_randReal_.generate<real>(d_random_, nRequiredRandNum);
    const real *d_randNum;

    real Tnorm = kT * beta;
    
    /* 1st */
    calculate_Jq(&d_Jq1_, d_J_, opNone, d_matq0_);
    d_randNum = d_randReal_.acquire<real>(N1_ * m_);
    tryFlipSA(&d_matq1_, d_Jq1_, N1_, m_, d_h1_, d_randNum, Tnorm);
    DEBUG_SYNC;

    /* 2nd */
    calculate_Jq(&d_Jq0_, d_J_, opTranspose, d_matq1_);
    d_randNum = d_randReal_.acquire<real>(N0_ * m_);
    tryFlipSA(&d_matq0_, d_Jq0_, N0_, m_, d_h0_, d_randNum, Tnorm);
    DEBUG_SYNC;
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::syncBits() {
    bitsPairX_.clear();
    bitsPairQ_.clear();

    devCopy_.cast(&h_q0_, d_matq0_);
    devCopy_.cast(&h_q1_, d_matq1_);
    devStream_->synchronize();

    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        BitSet q0(h_q0_.row(idx), N0_);
        BitSet q1(h_q1_.row(idx), N1_);
        bitsPairQ_.pushBack(BitSetPairArray::ValueType(q0, q1));
        BitSet x0 = x_from_q(q0), x1 = x_from_q(q1);
        bitsPairX_.pushBack(BitSetPairArray::ValueType(x0, x1));
    }
}


template class sqaod_cuda::CUDABipartiteGraphAnnealer<float>;
template class sqaod_cuda::CUDABipartiteGraphAnnealer<double>;
