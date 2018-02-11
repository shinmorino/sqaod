#include "CUDABipartiteGraphAnnealer.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>
#include "CUDAFormulas.h"


using namespace sqaod_cuda;
namespace sq = sqaod;

template<class real>
CUDABipartiteGraphAnnealer<real>::CUDABipartiteGraphAnnealer() {
    m_ = (SizeType)-1;
    annState_ = annNone;
}

template<class real>
CUDABipartiteGraphAnnealer<real>::CUDABipartiteGraphAnnealer(Device &device) {
    m_ = (SizeType)-1;
    annState_ = annNone;
    assignDevice(device);
}

template<class real>
CUDABipartiteGraphAnnealer<real>::~CUDABipartiteGraphAnnealer() {
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::assignDevice(Device &device) {
    devStream_ = device.defaultStream();
    devAlloc_ = device.objectAllocator();
    bgFuncs_.assignDevice(device);
    devCopy_.assignDevice(device);
    d_random_.assignDevice(device);
    d_randReal_.assignDevice(device);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::seed(unsigned long seed) {
    d_random_.seed(seed);
    annState_ |= annRandSeedGiven;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::getProblemSize(SizeType *N0, SizeType *N1, SizeType *m) const {
    *N0 = N0_;
    *N1 = N1_;
    *m = m_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::
setProblem(const HostVector &b0, const HostVector &b1, const HostMatrix &W, sq::OptimizeMethod om) {
    N0_ = W.cols;
    N1_ = W.rows;
    om_ = om;

    DeviceMatrix *d_W = devStream_->tempDeviceMatrix<real>(W.dim());
    DeviceVector *d_b0 = devStream_->tempDeviceVector<real>(b0.size);
    DeviceVector *d_b1 = devStream_->tempDeviceVector<real>(b1.size);
    devCopy_(d_W, W);
    if (om == sq::optMaximize)
        bgFuncs_.devMath.scale(d_W, -1., *d_W);
    bgFuncs_.calculate_hJc(&d_h0_, &d_h1_, &d_J_, &d_c_, *d_b0, *d_b1, *d_W);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::setNumTrotters(SizeType m) {
    throwErrorIf(m <= 0, "# trotters must be a positive integer.");
    m_ = m;
    devAlloc_->allocate(&d_matq0_, m_, N0_);
    devAlloc_->allocate(&d_matq1_, m_, N1_);

    HostObjectAllocator halloc;
    halloc.allocate(&h_E_, m_);
    annState_ |= annNTrottersGiven;
}

template<class real>
const BitsPairArray &CUDABipartiteGraphAnnealer<real>::get_x() const {
    return bitsPairX_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::set_x(const Bits &x0, const Bits &x1) {
    /* FIXME: add size check */
    HostVector rx0 = sq::x_to_q<real>(x0);
    HostVector rx1 = sq::x_to_q<real>(x1);
    DeviceVector *d_x0 = devStream_->tempDeviceVector<real>(rx0.size);
    DeviceVector *d_x1 = devStream_->tempDeviceVector<real>(rx1.size);
    devCopy_(d_x0, rx0);
    devCopy_(d_x1, rx1);
    bgFuncs_.devMath.scaleBroadcast(&d_matq0_, real(1.), *d_x0, opRowwise);
    bgFuncs_.devMath.scaleBroadcast(&d_matq1_, real(1.), *d_x1, opRowwise);
    annState_ |= annQSet;
}


template<class real>
const VectorType<real> CUDABipartiteGraphAnnealer<real>::get_E() const {
    return HostVector(h_E_.d_data, h_E_.size);
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::get_hJc(HostVector *h0, HostVector *h1,
                                               HostMatrix *J, real *c) const {
    devCopy_(h0, d_h0_);
    devCopy_(h1, d_h1_);
    devCopy_(J, d_J_);
    devCopy_(c, d_c_);
    devCopy_.synchronize();
}


template<class real>
const BitsPairArray &CUDABipartiteGraphAnnealer<real>::get_q() const {
    return bitsPairQ_;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::randomize_q() {
    cudaStream_t stream = devStream_->getCudaStream();
    sqaod_cuda::randomize_q(d_matq0_.d_data, d_random_, N0_, stream);
    sqaod_cuda::randomize_q(d_matq1_.d_data, d_random_, N1_, stream);
    annState_ |= annQSet;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::calculate_E() {
    bgFuncs_.calculate_E(&h_E_, d_h0_, d_h1_, d_J_, d_c_,
                         d_matq0_, d_matq1_);
    DeviceVector *d_E = devStream_->tempDeviceVector<real>(m_);
    real sign = (om_ == optMaximize) ? -1. : 1.;
    bgFuncs_.devMath.scale(&h_E_, sign, *d_E);
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::initAnneal() {
    if (!(annState_ & annRandSeedGiven))
        d_random_.seed();
    annState_ |= annRandSeedGiven;
    if (!(annState_ & annNTrottersGiven))
        setNumTrotters((N0_ + N1_) / 4);
    annState_ |= annNTrottersGiven;
    if (!(annState_ & annQSet))
        randomize_q();
    annState_ |= annQSet;
}

template<class real>
void CUDABipartiteGraphAnnealer<real>::finAnneal() {
    syncBits();
    calculate_E();
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
     *  bgFuncs_.devMath.matmul(d_Jq, d_J, op, qFixed, opTranspose);
     * Tranpose product for coalesced access. */
    op = (op == opNone) ? opTranspose : opNone;
    bgFuncs_.devMath.mmProduct(d_Jq, 1., d_qFixed, opNone, d_J, op);
}

template<int colorOffset, class real>
__global__ static void
tryFlipKernel(real *d_qAnneal, int N, int m, const real *d_Emat, const real *d_h,
              const real *d_realRand, real twoDivM, real coef, real invKT) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int iq = gidx;
    int im = 2 * gidy + colorOffset;
    
    if ((iq < N) && (im < m)) {
        real q = d_qAnneal[im * N + iq];
        real dE = - twoDivM * q * (d_h[iq] + d_Emat[N * im + iq]);

        int neibour0 = (im == 0) ? m - 1 : im - 1;
        int neibour1 = (im == m - 1) ? 0 : im + 1;
        dE -= q * (d_qAnneal[N * neibour0 + iq] + d_qAnneal[N * neibour1 + iq]) * coef;
        real thresh = dE < real(0.) ? real(1.) : exp(- dE * invKT); /* FIXME: check precision */
        if (thresh > d_realRand[N * gidy + iq])
            d_qAnneal[N * im + iq] = -q;
    }
}





template<class real> void CUDABipartiteGraphAnnealer<real>::
tryFlip(DeviceMatrix *d_qAnneal, const DeviceMatrix &d_Jq, int N, int m, int offset,
        const DeviceVector &d_h, const real *d_realRand, real G, real kT) {
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
    real invKT = real(1.) / kT;
    real twoDivM = real(2.) / m_;

    dim3 blockDim(64, 2);
    dim3 gridDim(divru((SizeType)N, blockDim.x), divru((SizeType)m, blockDim.y));
    if (offset == 0) {
        tryFlipKernel<0><<<gridDim, blockDim>>>
                (d_qAnneal->d_data, N, m, d_Jq.d_data, d_h.d_data, d_realRand, twoDivM, coef, invKT);
    }
    else {
        tryFlipKernel<1><<<gridDim, blockDim>>>
                (d_qAnneal->d_data, N, m, d_Jq.d_data, d_h.d_data, d_realRand, twoDivM, coef, invKT);
    }
    DEBUG_SYNC;
}


template<class real>
void CUDABipartiteGraphAnnealer<real>::annealOneStep(real G, real kT) {
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
    tryFlip(&d_matq1_, d_Jq1_, N1_, m_, 0, d_h1_, d_randNum, G, kT);
    /* 2nd */
    nTrottersToFlip = m_ / 2;
    d_randNum = d_randReal_.acquire<real>(N1_ * nTrottersToFlip);
    tryFlip(&d_matq1_, d_Jq1_, N1_, m_, 1, d_h1_, d_randNum, G, kT);
    /* 3rd */
    calculate_Jq(&d_Jq0_, d_J_, opTranspose, d_matq1_);
    nTrottersToFlip = (m_ + 1) / 2;
    d_randNum = d_randReal_.acquire<real>(N0_ * nTrottersToFlip);
    tryFlip(&d_matq0_, d_Jq0_, N0_, m_, 0, d_h0_, d_randNum, G, kT);
    /* 4th */
    nTrottersToFlip = m_ / 2;
    d_randNum = d_randReal_.acquire<real>(N0_ * nTrottersToFlip);
    tryFlip(&d_matq0_, d_Jq0_, N0_, m_, 1, d_h0_, d_randNum, G, kT);

}


template<class real>
void CUDABipartiteGraphAnnealer<real>::syncBits() {
    bitsPairX_.clear();
    bitsPairQ_.clear();

    

    HostObjectAllocator halloc;
    DeviceMatrix h_matq0, h_matq1;
    halloc.allocate(&h_matq0, d_matq0_.dim());
    halloc.allocate(&h_matq1, d_matq1_.dim());
    devCopy_(&h_matq0, d_matq0_);
    devCopy_(&h_matq1, d_matq1_);
    
    for (int idx = 0; idx < IdxType(m_); ++idx) {
        HostVector rq0(h_matq0.row(idx), N0_);
        HostVector rq1(h_matq1.row(idx), N1_);
        Bits q0 = sq::cast<char>(rq0);
        Bits q1 = sq::cast<char>(rq1);
        bitsPairQ_.pushBack(BitsPairArray::ValueType(q0, q1));
        Bits x0 = x_from_q(q0), x1 = x_from_q(q1);
        bitsPairX_.pushBack(BitsPairArray::ValueType(x0, x1));
    }

    halloc.deallocate(h_matq0);
    halloc.deallocate(h_matq1);
    
}


template class sqaod_cuda::CUDABipartiteGraphAnnealer<float>;
template class sqaod_cuda::CUDABipartiteGraphAnnealer<double>;
