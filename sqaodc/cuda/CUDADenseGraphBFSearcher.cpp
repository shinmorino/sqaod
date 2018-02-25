#include "CUDADenseGraphBFSearcher.h"
#include "Device.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <limits>

using namespace sqaod_cuda;

template<class real>
CUDADenseGraphBFSearcher<real>::CUDADenseGraphBFSearcher() {
    tileSize_ = 16384; /* FIXME: give a correct size */
    deviceAssigned_ = false;
}

template<class real>
CUDADenseGraphBFSearcher<real>::CUDADenseGraphBFSearcher(Device &device) {
    tileSize_ = 16384; /* FIXME: give a correct size */
    deviceAssigned_ = false;
    assignDevice(device);
}

template<class real>
CUDADenseGraphBFSearcher<real>::~CUDADenseGraphBFSearcher() {
}

template<class real>
void CUDADenseGraphBFSearcher<real>::deallocate() {
    if (h_packedXmin_.d_data != NULL)
        HostObjectAllocator().deallocate(h_packedXmin_);
    batchSearch_.deallocate();
}


template<class real>
void CUDADenseGraphBFSearcher<real>::assignDevice(Device &device) {
    throwErrorIf(deviceAssigned_, "Device already assigned.");
    batchSearch_.assignDevice(device);
    devCopy_.assignDevice(device);
    deviceAssigned_ = true;
}

template<class real>
void CUDADenseGraphBFSearcher<real>::setProblem(const Matrix &W, sq::OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    throwErrorIf(63 < N_, "N must be smaller than 64, N=%d.", N_);
    throwErrorIf(!deviceAssigned_, "Device not set.");
    N_ = W.rows;
    W_ = W;
    om_ = om;
    if (om_ == sq::optMaximize)
        W_ *= real(-1.);

    setState(solProblemSet);
}

template<class real>
const sq::BitsArray &CUDADenseGraphBFSearcher<real>::get_x() const {
    throwErrorIfSolutionNotAvailable();
    return xList_;
}

template<class real>
const sq::VectorType<real> &CUDADenseGraphBFSearcher<real>::get_E() const {
    throwErrorIfSolutionNotAvailable();
    return E_;
}

template<class real>
void CUDADenseGraphBFSearcher<real>::initSearch() {
    throwErrorIfProblemNotSet();
    if (isInitialized())
        deallocate();
    
    sq::SizeType maxTileSize = 1u << N_;
    if (maxTileSize < tileSize_) {
        tileSize_ = maxTileSize;
        sq::log("Tile size is adjusted to %d for N=%d", maxTileSize, N_);
    }
    batchSearch_.setProblem(W_, tileSize_);
    HostObjectAllocator().allocate(&h_packedXmin_, tileSize_);

    Emin_ = std::numeric_limits<real>::max();
    xList_.clear();
    xMax_ = 1ull << N_;

    setState(solInitialized);
}


template<class real>
void CUDADenseGraphBFSearcher<real>::finSearch() {
    throwErrorIfNotInitialized();

    batchSearch_.synchronize();
    const DevicePackedBitsArray &dPackedXmin = batchSearch_.get_xMins();
    sq::SizeType nXMin = std::min(tileSize_, dPackedXmin.size);
    devCopy_(&h_packedXmin_, dPackedXmin);
    devCopy_.synchronize();
    
    xList_.clear();
    E_.resize(nXMin);
    E_ = Emin_;
    if (om_ == sq::optMaximize)
        E_ *= real(-1.);
    for (sq::IdxType idx = 0; idx < (sq::IdxType)nXMin; ++idx) {
        sq::Bits bits;
        unpackBits(&bits, h_packedXmin_[idx], N_);
        xList_.pushBack(bits); // FIXME: apply move
    }

    setState(solSolutionAvailable);
}

template<class real>
void CUDADenseGraphBFSearcher<real>::searchRange(sq::PackedBits xBegin, sq::PackedBits xEnd) {
    throwErrorIfNotInitialized();

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


template class sqaod_cuda::CUDADenseGraphBFSearcher<float>;
template class sqaod_cuda::CUDADenseGraphBFSearcher<double>;
