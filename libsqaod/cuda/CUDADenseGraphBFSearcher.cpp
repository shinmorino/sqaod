#include "CUDADenseGraphBFSearcher.h"
#include "Device.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <limits>


using namespace sqaod_cuda;
namespace sq = sqaod;

template<class real>
CUDADenseGraphBFSearcher<real>::CUDADenseGraphBFSearcher() {
    tileSize_ = 16384; /* FIXME: give a correct size */
}

template<class real>
CUDADenseGraphBFSearcher<real>::CUDADenseGraphBFSearcher(Device &device) {
    tileSize_ = 16384; /* FIXME: give a correct size */
    assignDevice(device);
}

template<class real>
CUDADenseGraphBFSearcher<real>::~CUDADenseGraphBFSearcher() {
}

template<class real>
void CUDADenseGraphBFSearcher<real>::assignDevice(Device &device) {
    batchSearch_.assignDevice(device);
    devCopy_.assignDevice(device);
}

template<class real>
void CUDADenseGraphBFSearcher<real>::getProblemSize(sqaod::SizeType *N) const {
    *N = N_;
}

template<class real>
void CUDADenseGraphBFSearcher<real>::setProblem(const Matrix &W, sq::OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    N_ = W.rows;
    W_ = W;
    om_ = om;
    if (om_ == sq::optMaximize)
        W_ *= real(-1.);
}

template<class real>
void CUDADenseGraphBFSearcher<real>::setTileSize(sqaod::SizeType tileSize) {
    tileSize_ = tileSize;
}

template<class real>
const sq::BitsArray &CUDADenseGraphBFSearcher<real>::get_x() const {
    return xList_;
}

template<class real>
const sq::VectorType<real> &CUDADenseGraphBFSearcher<real>::get_E() const {
    return E_;
}

template<class real>
void CUDADenseGraphBFSearcher<real>::initSearch() {
    batchSearch_.setProblem(W_, tileSize_);
    HostObjectAllocator().allocate(&h_packedXmin_, tileSize_);

    Emin_ = std::numeric_limits<real>::max();
    xList_.clear();
    xMax_ = 1ull << N_;
}


template<class real>
void CUDADenseGraphBFSearcher<real>::finSearch() {
    batchSearch_.synchronize();
    const DevicePackedBitsArray &dPackedXmin = batchSearch_.get_xMins();
    sqaod::SizeType nXMin = std::min(tileSize_, dPackedXmin.size);
    devCopy_(&h_packedXmin_, dPackedXmin);
    devCopy_.synchronize();
    
    xList_.clear();
    E_.resize(nXMin);
    E_ = Emin_;
    if (om_ == sq::optMaximize)
        E_ *= real(-1.);
    for (sqaod::IdxType idx = 0; idx < (sqaod::IdxType)nXMin; ++idx) {
        sq::Bits bits;
        unpackBits(&bits, h_packedXmin_[idx], N_);
        xList_.pushBack(bits); // FIXME: apply move
    }
}

template<class real>
void CUDADenseGraphBFSearcher<real>::searchRange(sq::PackedBits xBegin, sq::PackedBits xEnd) {
    /* FIXME: Use multiple searchers, multi GPU */
    throwErrorIf(xBegin > xEnd, "xBegin should be larger than xEnd");
    xBegin = std::min(std::max(0ULL, xBegin), xMax_);
    xEnd = std::min(std::max(0ULL, xEnd), xMax_);
    if (xBegin == xEnd)
        return; /* Nothing to do */

    batchSearch_.calculate_E(xBegin, xEnd);
    batchSearch_.synchronize();

    real newEmin = batchSearch_.get_Emin();
    if (newEmin < Emin_) {
        batchSearch_.partition_xMins(false);
        Emin_ = newEmin;
    }
    else if (newEmin == Emin_) {
        batchSearch_.partition_xMins(true);
    }
}


template<class real>
void CUDADenseGraphBFSearcher<real>::search() {
    initSearch();
    int iStep = (int)std::min((unsigned long long)tileSize_, xMax_);
    for (sq::PackedBits iTile = 0; iTile < xMax_; iTile += iStep) {
        searchRange(iTile, iTile + iStep);
    }
    finSearch();
}

template class sqaod_cuda::CUDADenseGraphBFSearcher<float>;
template class sqaod_cuda::CUDADenseGraphBFSearcher<double>;
