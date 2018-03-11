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
    clearState(solProblemSet);

    N_ = W.rows;
    W_ = W;
    om_ = om;
    if (om_ == sq::optMaximize)
        W_ *= real(-1.);

    setState(solProblemSet);
}

template<class real>
sq::Preferences CUDADenseGraphBFSearcher<real>::getPreferences() const {
    sq::Preferences prefs = Base::getPreferences();
    prefs.pushBack(sq::Preference(sq::pnDevice, "cuda"));
    return prefs;
}

template<class real>
const sq::BitSetArray &CUDADenseGraphBFSearcher<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution(); /* synchronized there */
    return xList_;
}

template<class real>
const sq::VectorType<real> &CUDADenseGraphBFSearcher<real>::get_E() const {
    if (!isEAvailable())
        const_cast<This*>(this)->calculate_E();
    return E_;
}

template<class real>
void CUDADenseGraphBFSearcher<real>::prepare() {
    throwErrorIfProblemNotSet();
    deallocate();
    
    x_ = 0;
    sq::SizeType maxTileSize = 1u << N_;
    if (maxTileSize < tileSize_) {
        tileSize_ = maxTileSize;
        sq::log("Tile size is adjusted to %d for N=%d", maxTileSize, N_);
    }
    batchSearch_.setProblem(W_, tileSize_);
    HostObjectAllocator().allocate(&h_packedXmin_, tileSize_ * 2);

    Emin_ = std::numeric_limits<real>::max();
    xList_.clear();
    xMax_ = 1ull << N_;

    setState(solPrepared);
}


template<class real>
void CUDADenseGraphBFSearcher<real>::calculate_E() {
    throwErrorIfNotPrepared();
    if (xList_.empty())
        E_.resize(1);
    else
        E_.resize(xList_.size());
    E_ = Emin_;
    if (om_ == sq::optMaximize)
        E_ *= real(-1.);

    setState(solEAvailable);
}


template<class real>
void CUDADenseGraphBFSearcher<real>::makeSolution() {
    throwErrorIfNotPrepared();

    batchSearch_.synchronize();
    const DevicePackedBitSetArray &dPackedXmin = batchSearch_.get_xMins();
    sq::SizeType nXMin = std::min(tileSize_, dPackedXmin.size);
    devCopy_(&h_packedXmin_, dPackedXmin);
    devCopy_.synchronize();
    
    xList_.clear();
    for (sq::IdxType idx = 0; idx < (sq::IdxType)nXMin; ++idx) {
        sq::BitSet bits;
        unpackBitSet(&bits, h_packedXmin_[idx], N_);
        xList_.pushBack(bits); // FIXME: apply move
    }
    setState(solSolutionAvailable);
    calculate_E();
}

template<class real>
bool CUDADenseGraphBFSearcher<real>::searchRange(sq::PackedBitSet *curXEnd) {
    throwErrorIfNotPrepared();
    clearState(solSolutionAvailable);

    /* FIXME: Use multiple searchers, multi GPU */

    sq::PackedBitSet xBegin = x_;
    sq::PackedBitSet xEnd = std::min(x_ + tileSize_, xMax_);
    if (xBegin < xEnd) {
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
    
    x_ = xEnd;
    if (curXEnd != NULL)
        *curXEnd = x_;
    return x_ == xMax_;
}


template class sqaod_cuda::CUDADenseGraphBFSearcher<float>;
template class sqaod_cuda::CUDADenseGraphBFSearcher<double>;
