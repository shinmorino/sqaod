#include "CUDADenseGraphBFSolver.h"
#include "CUDAFormulas.h"
#include <cmath>

#include <float.h>
#include <algorithm>
#include <limits>


using namespace sqaod_cuda;
using namespace sqaod;

template<class real>
CUDADenseGraphBFSolver<real>::CUDADenseGraphBFSolver() {
    tileSize_ = 4096; /* FIXME: give a correct size */
}

template<class real>
CUDADenseGraphBFSolver<real>::~CUDADenseGraphBFSolver() {
}


template<class real>
void CUDADenseGraphBFSolver<real>::getProblemSize(int *N) const {
    *N = N_;
}

template<class real>
void CUDADenseGraphBFSolver<real>::setProblem(const Matrix &W, OptimizeMethod om) {
    THROW_IF(!isSymmetric(W), "W is not symmetric.");
    N_ = W.rows;
    W_ = W;
    om_ = om;
    if (om_ == optMaximize)
        W_.map().array() *= real(-1.);
}

template<class real>
void CUDADenseGraphBFSolver<real>::setTileSize(int tileSize) {
    tileSize_ = tileSize;
}

template<class real>
const BitsArray &CUDADenseGraphBFSolver<real>::get_x() const {
    return xList_;
}

template<class real>
const VectorType<real> &CUDADenseGraphBFSolver<real>::get_E() const {
    return E_;
}

template<class real>
void CUDADenseGraphBFSolver<real>::initSearch() {
    Emin_ = std::numeric_limits<real>::max();
    xList_.clear();
    xMax_ = 1 << N_;
}


template<class real>
void CUDADenseGraphBFSolver<real>::finSearch() {
    xList_.clear();
    devCopy_(&packedXmin_, d_xMin_);
    stream_->synchronize();
    E_.resize(packedXmin_.size());
    E_.mapToRowVector().array() = (om_ == optMaximize) ? - Emin_ : Emin_;
    for (size_t idx = 0; idx < packedXmin_.size(); ++idx) {
        Bits bits;
        unpackBits(&bits, packedXmin_[idx], N_);
        xList_.pushBack(bits); // FIXME: apply move
    }
}


template<class real>
void CUDADenseGraphBFSolver<real>::batchCalculate_E(unsigned long long iBegin, unsigned long long iEnd) {
    iBegin = std::min(std::max(0ULL, iBegin), xMax_);
    iEnd = std::min(std::max(0ULL, iEnd), xMax_);
    batchSearch_.calculate_E(iBegin, iEnd);
}

template<class real>
void CUDADenseGraphBFSolver<real>::updateXmins() {
    /* FIXME: delayed copy */
    if (batchSearch_.get_Emin() < Emin_) {
        batchSearch_.partition_xMin();
        batchSearch_.sync(); // ToDo: remove sync
        DevicePackedBitsArray::swap(&d_xMin_, batchSearch_.get_xMin());
    }
    if (batchSearch_.get_Emin() == Emin_) {
        batchSearch_.partition_xMin();
        batchSearch_.sync(); // ToDo: remove sync
        d_xMin_.append(*batchSearch_.get_xMin());
    }
}


template<class real>
void CUDADenseGraphBFSolver<real>::search() {
    int iStep = (int)std::min((unsigned long long)tileSize_, xMax_);

    initSearch();
    for (unsigned long long iTile = 0; iTile < xMax_; iTile += iStep) {
        batchCalculate_E(iTile, iTile + iStep);
        updateXmins();
    }
    finSearch();
}

template class sqaod_cuda::CUDADenseGraphBFSolver<float>;
template class sqaod_cuda::CUDADenseGraphBFSolver<double>;
