#include "CUDABipartiteGraphAnnealer.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>

using namespace sqaod_cuda;

template<class real>
CUDABipartiteGraphAnnealer<real>::CUDABipartiteGraphAnnealer() {
    devStream_ = NULL;
    m_ = (SizeType)-1;
}

template<class real>
CUDABipartiteGraphAnnealer<real>::CUDABipartiteGraphAnnealer(Device &device) {
    devStream_ = NULL;
    m_ = (SizeType)-1;
    assignDevice(device);
}

template<class real>
CUDABipartiteGraphAnnealer<real>::~CUDABipartiteGraphAnnealer() {
    if (isInitialized())
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
    
    clearState(solInitialized);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::deallocate() {
    if (isProblemSet())
        deallocateInternalObjects();
    if (isInitialized())
        deallocateInternalObjects();
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
    return sq::algoColoring;
}

template<class real>
sq::Algorithm CUDABipartiteGraphAnnealer<real>::getAlgorithm() const {
    return sq::algoColoring;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::seed(unsigned int seed) {
    throwErrorIf(devStream_ == NULL, "Device not set.");
    d_random_.seed(seed);
    setState(solRandSeedGiven);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::
setProblem(const HostVector &b0, const HostVector &b1, const HostMatrix &W, sq::OptimizeMethod om) {
    /* FIXME: add QUBO dim check. */
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

    devFormulas_.calculate_hJc(&d_h0_, &d_h1_, &d_J_, &d_c_, *d_b0, *d_b1, *d_W);

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
    throwErrorIfSolutionNotAvailable();
    return E_;
}

template<class real>
const sq::BitsPairArray &CUDABipartiteGraphAnnealer<real>::get_x() const {
    throwErrorIfSolutionNotAvailable();
    return bitsPairX_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::set_x(const Bits &x0, const Bits &x1) {
    throwErrorIfNotInitialized();
    throwErrorIf(x0.size != N0_,
                 "Dimension of x0, %d,  should be equal to N0, %d.", x0.size, N0_);
    throwErrorIf(x1.size != N1_,
                 "Dimension of x1, %d,  should be equal to N1, %d.", x1.size, N1_);

    HostVector rx0 = sq::x_to_q<real>(x0);
    HostVector rx1 = sq::x_to_q<real>(x1);
    DeviceVector *d_x0 = devStream_->tempDeviceVector<real>(rx0.size);
    DeviceVector *d_x1 = devStream_->tempDeviceVector<real>(rx1.size);
    devCopy_(d_x0, rx0);
    devCopy_(d_x1, rx1);
    devFormulas_.devMath.scaleBroadcast(&d_matq0_, real(1.), *d_x0, opRowwise);
    devFormulas_.devMath.scaleBroadcast(&d_matq1_, real(1.), *d_x1, opRowwise);
    setState(solQSet);
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::get_hJc(HostVector *h0, HostVector *h1,
                                               HostMatrix *J, real *c) const {
    throwErrorIfProblemNotSet();

    devCopy_(h0, d_h0_);
    devCopy_(h1, d_h1_);
    devCopy_(J, d_J_);
    devCopy_(c, d_c_);
    devCopy_.synchronize();
}


template<class real>
const sq::BitsPairArray &CUDABipartiteGraphAnnealer<real>::get_q() const {
    return bitsPairQ_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::randomize_q() {
    throwErrorIfNotInitialized();

    cudaStream_t stream = devStream_->getCudaStream();
    sqaod_cuda::randomize_q(d_matq0_.d_data, d_random_, N0_ * m_, stream);
    sqaod_cuda::randomize_q(d_matq1_.d_data, d_random_, N1_ * m_, stream);
    setState(solQSet);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::calculate_E() {
    throwErrorIfQNotSet();

    DeviceVector *d_E = devStream_->tempDeviceVector<real>(m_);
    devFormulas_.calculate_E(d_E, d_h0_, d_h1_, d_J_, d_c_,
                             d_matq0_, d_matq1_);
    real sign = (om_ == sq::optMaximize) ? -1. : 1.;
    devFormulas_.devMath.scale(&h_E_, sign, *d_E);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::initAnneal() {
    throwErrorIfProblemNotSet();

    if (!isRandSeedGiven())
        d_random_.seed();
    setState(solRandSeedGiven);

    if (isInitialized())
        deallocateInternalObjects();

    devAlloc_->allocate(&d_matq0_, m_, N0_);
    devAlloc_->allocate(&d_matq1_, m_, N1_);

    HostObjectAllocator halloc;
    halloc.allocate(&h_E_, m_);
    E_.map(h_E_.d_data, h_E_.size);
    halloc.allocate(&h_q0_, m_, N0_);
    halloc.allocate(&h_q1_, m_, N1_);
    bitsPairX_.reserve(m_);
    bitsPairQ_.reserve(m_);

    int requiredSize = ((N0_ + N1_) * m_ * (nRunsPerRandGen + 1)) * sizeof(real) / 4;
    d_random_.setRequiredSize(requiredSize);
    setState(solInitialized);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::finAnneal() {
    throwErrorIfQNotSet();

    syncBits();
    calculate_E();
    devStream_->synchronize();

    setState(solSolutionAvailable);
}



// template<class real>
// void CUDABipartiteGraphAnnealer<real>::
// annealHalfStep(DeviceMatrix *d_qAnneal, int N,
//                const DeviceVector &d_h, real G, real kT) {
//     real twoDivM = real(2.) / m_;
//     bgFuncs_.devMath.matmul(d_Jq, d_J, op, qFixed, opTranspose);
//     real tempCoef = std::log(std::tanh(G / kT / m_)) / kT;
//     real invKT = real(1.) / kT;
//     for (int loop = 0; loop < IdxType(N * m_); ++loop) {
//         real q = qAnneal(im, iq);
//         real dE = - twoDivM * q * (h[iq] + dEmat(iq, im));
//         int mNeibour0 = (im + m_ - 1) % m_;
//         int mNeibour1 = (im + 1) % m_;
//         dE -= q * (qAnneal(mNeibour0, iq) + qAnneal(mNeibour1, iq)) * tempCoef;
//         real thresh = dE < real(0.) ? real(1.) : std::exp(- dE * invKT);
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

template<class real>
__global__ static void
tryFlipKernel(real *d_qAnneal, int N, int m, int colorOffset, const real *d_Emat, const real *d_h,
              const real *d_realRand, real twoDivM, real coef, real invKT) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int iq = gidx;
    int im = 2 * gidy + colorOffset;
    
    if ((iq < N) && (im < m)) {
        real q = d_qAnneal[im * N + iq];
        real dE = - twoDivM * q * (d_h[iq] + d_Emat[im * N + iq]);

        int neibour0 = (im == 0) ? m - 1 : im - 1;
        int neibour1 = (im == m - 1) ? 0 : im + 1;
        dE -= q * (d_qAnneal[neibour0 * N + iq] + d_qAnneal[neibour1 * N + iq]) * coef;
        real thresh = dE < real(0.) ? real(1.) : exp(- dE * invKT); /* FIXME: check precision */
        if (thresh > d_realRand[N * gidy + iq])
            d_qAnneal[im * N + iq] = - q;
    }
}





template<class real> void CUDABipartiteGraphAnnealer<real>::
tryFlip(DeviceMatrix *d_qAnneal, const DeviceMatrix &d_Jq, int N, int m,
    int nTrottersToFlip, int offset, 
    const DeviceVector &d_h, const real *d_realRand, real G, real kT) {
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
    real invKT = real(1.) / kT;
    real twoDivM = real(2.) / m_;

    dim3 blockDim(64, 2);
    dim3 gridDim(divru((SizeType)N, blockDim.x), divru((SizeType)nTrottersToFlip, blockDim.y));
    if (offset == 0) {
        tryFlipKernel<<<gridDim, blockDim>>>
                (d_qAnneal->d_data, N, m, 0, d_Jq.d_data, d_h.d_data, d_realRand, twoDivM, coef, invKT);
    }
    else {
        tryFlipKernel<<<gridDim, blockDim>>>
                (d_qAnneal->d_data, N, m, 1, d_Jq.d_data, d_h.d_data, d_realRand, twoDivM, coef, invKT);
    }
    DEBUG_SYNC;
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::annealOneStep(real G, real kT) {
    throwErrorIfQNotSet();

    int nRequiredRandNum = (N0_ + N1_) * m_;
    if (!d_randReal_.available(nRequiredRandNum))
        d_randReal_.generate<real>(d_random_, nRequiredRandNum);
    const real *d_randNum;
    int nTrottersToFlip;

    /* FIXME: consider Jq to use half trotters. */
    
    /* 1st */
    calculate_Jq(&d_Jq1_, d_J_, opNone, d_matq0_);
    nTrottersToFlip = (m_ + 1) / 2;
    d_randNum = d_randReal_.acquire<real>(N1_ * nTrottersToFlip);
    tryFlip(&d_matq1_, d_Jq1_, N1_, m_, nTrottersToFlip, 0, d_h1_, d_randNum, G, kT);
    DEBUG_SYNC;
    /* 2nd */
    nTrottersToFlip = m_ / 2;
    d_randNum = d_randReal_.acquire<real>(N1_ * nTrottersToFlip);
    tryFlip(&d_matq1_, d_Jq1_, N1_, m_, nTrottersToFlip, 1, d_h1_, d_randNum, G, kT);
    DEBUG_SYNC;
    /* 3rd */
    calculate_Jq(&d_Jq0_, d_J_, opTranspose, d_matq1_);
    nTrottersToFlip = (m_ + 1) / 2;
    d_randNum = d_randReal_.acquire<real>(N0_ * nTrottersToFlip);
    tryFlip(&d_matq0_, d_Jq0_, N0_, m_, nTrottersToFlip, 0, d_h0_, d_randNum, G, kT);
    DEBUG_SYNC;
    /* 4th */
    nTrottersToFlip = m_ / 2;
    d_randNum = d_randReal_.acquire<real>(N0_ * nTrottersToFlip);
    tryFlip(&d_matq0_, d_Jq0_, N0_, m_, nTrottersToFlip, 1, d_h0_, d_randNum, G, kT);
    DEBUG_SYNC;
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::syncBits() {
    bitsPairX_.clear();
    bitsPairQ_.clear();

    devFormulas_.devMath.toBits(&h_q0_, d_matq0_);
    devFormulas_.devMath.toBits(&h_q1_, d_matq1_);
    devStream_->synchronize();

    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        Bits q0(h_q0_.row(idx), N0_);
        Bits q1(h_q1_.row(idx), N1_);
        bitsPairQ_.pushBack(BitsPairArray::ValueType(q0, q1));
        Bits x0 = x_from_q(q0), x1 = x_from_q(q1);
        bitsPairX_.pushBack(BitsPairArray::ValueType(x0, x1));
    }
}


template class sqaod_cuda::CUDABipartiteGraphAnnealer<float>;
template class sqaod_cuda::CUDABipartiteGraphAnnealer<double>;
