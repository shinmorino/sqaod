#include "CUDADenseGraphBFSolver.h"
#include "CUDAFormulas.h"
#include "Device.h"
#include <cmath>

#include <float.h>
#include <algorithm>
#include <limits>


using namespace sqaod_cuda;
namespace sq = sqaod;

template<class real>
CUDADenseGraphBFSolver<real>::CUDADenseGraphBFSolver() {
    tileSize_ = 4096; /* FIXME: give a correct size */
}

template<class real>
CUDADenseGraphBFSolver<real>::~CUDADenseGraphBFSolver() {
}


template<class real>
void CUDADenseGraphBFSolver<real>::assignDevice(Device &device) {
    batchSearch_.assignDevice(device);
}

template<class real>
void CUDADenseGraphBFSolver<real>::getProblemSize(sqaod::SizeType *N) const {
    *N = N_;
}

template<class real>
void CUDADenseGraphBFSolver<real>::setProblem(const Matrix &W, sq::OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    N_ = W.rows;
    W_ = W;
    om_ = om;
    if (om_ == sq::optMaximize)
        W_.map().array() *= real(-1.);
}

template<class real>
void CUDADenseGraphBFSolver<real>::setTileSize(sqaod::SizeType tileSize) {
    tileSize_ = tileSize;
}

template<class real>
const sq::BitsArray &CUDADenseGraphBFSolver<real>::get_x() const {
    return xList_;
}

template<class real>
const sq::VectorType<real> &CUDADenseGraphBFSolver<real>::get_E() const {
    return E_;
}

template<class real>
void CUDADenseGraphBFSolver<real>::initSearch() {
    batchSearch_.setProblem(W_, tileSize_);
    Emin_ = std::numeric_limits<real>::max();
    xList_.clear();
    xMax_ = 1 << N_;
}


template<class real>
void CUDADenseGraphBFSolver<real>::finSearch() {
    batchSearch_.synchronize();
    const PackedBitsArray &packedXmin = batchSearch_.get_xMins();
    sqaod::SizeType nXMin = std::min(tileSize_, (sqaod::SizeType)packedXmin.size());
    
    xList_.clear();
    E_.resize(nXMin);
    E_.mapToRowVector().array() = (om_ == sq::optMaximize) ? - Emin_ : Emin_;
    for (size_t idx = 0; idx < nXMin; ++idx) {
        sq::Bits bits;
        unpackBits(&bits, packedXmin[idx], N_);
        xList_.pushBack(bits); // FIXME: apply move
    }
}


template<class real>
void CUDADenseGraphBFSolver<real>::search() {
    int iStep = (int)std::min((unsigned long long)tileSize_, xMax_);

    initSearch();
    for (sq::PackedBits iTile = 0; iTile < xMax_; iTile += iStep) {
        /* FIXME: Use multiple searchers, multi GPU */
        sq::PackedBits iBegin = std::min(std::max(0ULL, iTile), xMax_);
        sq::PackedBits iEnd = std::min(std::max(0ULL, iTile + iStep), xMax_);
        batchSearch_.calculate_E(iBegin, iEnd);
        batchSearch_.synchronize();
        real newEmin = batchSearch_.get_Emin();
        if (newEmin < Emin_) {
            batchSearch_.partition_xMins(false);
            Emin_ = newEmin;
        }
        else if (batchSearch_.get_Emin() == Emin_) {
            batchSearch_.partition_xMins(true);
        }
    }
    finSearch();
}

template class sqaod_cuda::CUDADenseGraphBFSolver<float>;
template class sqaod_cuda::CUDADenseGraphBFSolver<double>;
