#include "CUDABipartiteGraphBFSolver.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>

using namespace sqaod_cuda;
namespace sq = sqaod;

template<class real>
CUDABipartiteGraphBFSolver<real>::CUDABipartiteGraphBFSolver() {
    tileSize0_ = 1024;
    tileSize1_ = 1024;
}

template<class real>
CUDABipartiteGraphBFSolver<real>::CUDABipartiteGraphBFSolver(Device &device) {
    tileSize0_ = 1024;
    tileSize1_ = 1024;
    assignDevice(device);
}

template<class real>
CUDABipartiteGraphBFSolver<real>::~CUDABipartiteGraphBFSolver() {
}

template<class real>
void CUDABipartiteGraphBFSolver<real>::assignDevice(Device &device) {
    batchSearch_.assignDevice(device, device.defaultStream());
}


template<class real>
void CUDABipartiteGraphBFSolver<real>::getProblemSize(int *N0, int *N1) const {
    *N0 = N0_;
    *N1 = N1_;
}

template<class real>
void CUDABipartiteGraphBFSolver<real>::setProblem(const HostVector &b0, const HostVector &b1,
                                                  const HostMatrix &W, sqaod::OptimizeMethod om) {
    N0_ = b0.size;
    N1_ = b1.size;
    b0_ = b0;
    b1_ = b1;
    W_ = W;
    om_ = om;
    if (om_ == sq::optMaximize) {
        W_ *= real(-1.);
        b0_ *= real(-1.);
        b1_ *= real(-1.);
    }
}

template<class real>
void CUDABipartiteGraphBFSolver<real>::setTileSize(SizeType tileSize0, SizeType tileSize1) {
    tileSize0_ = tileSize0;
    tileSize1_ = tileSize1;
}

template<class real>
const sq::BitsPairArray &CUDABipartiteGraphBFSolver<real>::get_x() const {
    return minXPairs_;
}

template<class real>
const sq::VectorType<real> &CUDABipartiteGraphBFSolver<real>::get_E() const {
    return E_;
}

template<class real>
void CUDABipartiteGraphBFSolver<real>::initSearch() {
    Emin_ = std::numeric_limits<real>::max();
    xPackedPairs_.clear();
    x0max_ = 1ull << N0_;
    x1max_ = 1ull << N1_;
    /* FIXME: create persistent tileSize member. */
    tileSize0_ = (sq::SizeType)std::min((sq::PackedBits)tileSize0_, x0max_);
    tileSize1_ = (sq::SizeType)std::min((sq::PackedBits)tileSize1_, x1max_);
    batchSearch_.setProblem(b0_, b1_, W_, tileSize0_, tileSize1_);
    SizeType minXPairsSize = (tileSize0_ * tileSize1_) * 2;
    HostObjectAllocator().allocate(&h_packedMinXPairs_, minXPairsSize);

}

template<class real>
void CUDABipartiteGraphBFSolver<real>::finSearch() {
    batchSearch_.synchronize();
    const DevicePackedBitsPairArray &dPackedXminPairs = batchSearch_.get_minXPairs();
    SizeType nXMin = dPackedXminPairs.size;
    devCopy_(&h_packedMinXPairs_, dPackedXminPairs);
    devCopy_.synchronize();
    
    minXPairs_.clear();
    E_.resize(nXMin);
    E_ = Emin_;
    if (om_ == sq::optMaximize)
        E_ *= real(-1.);
    for (sqaod::IdxType idx = 0; idx < (sqaod::IdxType)nXMin; ++idx) {
        sq::Bits bits0, bits1;
        unpackBits(&bits0, h_packedMinXPairs_[idx].bits0, N0_);
        unpackBits(&bits1, h_packedMinXPairs_[idx].bits1, N1_);
        minXPairs_.pushBack(BitsPairArray::ValueType(bits0, bits1)); // FIXME: apply move
    }
}

template<class real>
void CUDABipartiteGraphBFSolver<real>::searchRange(PackedBits xBegin0, PackedBits xEnd0,
                                                   PackedBits xBegin1, PackedBits xEnd1) {
    /* FIXME: Use multiple searchers, multi GPU */
    throwErrorIf(xBegin0 > xEnd0, "xBegin0 should be larger than xEnd0");
    throwErrorIf(xBegin1 > xEnd1, "xBegin1 should be larger than xEnd1");
    if ((xBegin0 == xEnd0) || (xBegin1 == xEnd1))
        return; /* Nothing to do */
    xBegin0 = std::min(std::max(0ULL, xBegin0), x0max_);
    xEnd0 = std::min(std::max(0ULL, xEnd0), x0max_);
    xBegin1 = std::min(std::max(0ULL, xBegin1), x1max_);
    xEnd1 = std::min(std::max(0ULL, xEnd1), x1max_);
    
    batchSearch_.calculate_E(xBegin0, xEnd0, xBegin1, xEnd1);
    batchSearch_.synchronize();

    real newEmin = batchSearch_.get_Emin();
    if (newEmin < Emin_) {
        batchSearch_.partition_minXPairs(false);
        Emin_ = newEmin;
    }
    else if (newEmin == Emin_) {
        batchSearch_.partition_minXPairs(true);
    }
    /* FIXME: add max limits of # min vectors. */
}

template<class real>
void CUDABipartiteGraphBFSolver<real>::search() {
    initSearch();

    PackedBits iStep0 = std::min((PackedBits)tileSize0_, x0max_);
    PackedBits iStep1 = std::min((PackedBits)tileSize1_, x1max_);
    for (PackedBits iTile1 = 0; iTile1 < x1max_; iTile1 += iStep1) {
        for (PackedBits iTile0 = 0; iTile0 < x0max_; iTile0 += iStep0) {
            searchRange(iTile0, iTile0 + iStep0, iTile1, iTile1 + iStep1);
        }
    }
    finSearch();
}

template class sqaod_cuda::CUDABipartiteGraphBFSolver<float>;
template class sqaod_cuda::CUDABipartiteGraphBFSolver<double>;
