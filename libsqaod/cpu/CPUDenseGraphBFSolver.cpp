#include "CPUDenseGraphBFSolver.h"
#include "CPUFormulas.h"
#include <cmath>

#include <float.h>
#include <algorithm>

using namespace sqaod;

template<class real>
CPUDenseGraphBFSolver<real>::CPUDenseGraphBFSolver() {
    tileSize_ = 1024;
}

template<class real>
CPUDenseGraphBFSolver<real>::~CPUDenseGraphBFSolver() {
}


template<class real>
void CPUDenseGraphBFSolver<real>::seed(unsigned long seed) {
    random_.seed(seed);
}

template<class real>
void CPUDenseGraphBFSolver<real>::getProblemSize(int *N) const {
    *N = N_;
}

template<class real>
void CPUDenseGraphBFSolver<real>::setProblem(const Matrix &W, OptimizeMethod om) {
    THROW_IF(!isSymmetric(W), "W is not symmetric.");
    N_ = W.rows;
    W_ = W.map();
    om_ = om;
    if (om_ == optMaximize)
        W_ *= real(-1.);
}

template<class real>
void CPUDenseGraphBFSolver<real>::setTileSize(int tileSize) {
    tileSize_ = tileSize;
}

template<class real>
const BitsArray &CPUDenseGraphBFSolver<real>::get_x() const {
    return xList_;
}

template<class real>
const VectorType<real> &CPUDenseGraphBFSolver<real>::get_E() const {
    return E_;
}

template<class real>
void CPUDenseGraphBFSolver<real>::initSearch() {
    minE_ = FLT_MAX;
    packedXList_.clear();
    xList_.clear();
    xMax_ = 1ull << N_;
}


template<class real>
void CPUDenseGraphBFSolver<real>::finSearch() {
    xList_.clear();
    for (int idx = 0; idx < packedXList_.size(); ++idx) {
        Bits bits;
        unpackBits(&bits, packedXList_[idx], N_);
        xList_.pushBack(bits);
    }
    E_.resize((SizeType)packedXList_.size());
    E_.mapToRowVector().array() = (om_ == optMaximize) ? - minE_ : minE_;
}


template<class real>
void CPUDenseGraphBFSolver<real>::searchRange(unsigned long long iBegin, unsigned long long iEnd) {
    iBegin = std::min(std::max(0ULL, iBegin), xMax_);
    iEnd = std::min(std::max(0ULL, iEnd), xMax_);
    DGFuncs<real>::batchSearch(&minE_, &packedXList_, W_, iBegin, iEnd);
    /* FIXME: add max limits of # min vectors. */
}

template<class real>
void CPUDenseGraphBFSolver<real>::search() {
    int iStep = (int)std::min((unsigned long long)tileSize_, xMax_);

    initSearch();
    for (unsigned long long iTile = 0; iTile < xMax_; iTile += iStep) {
        searchRange(iTile, iTile + iStep);
    }
    finSearch();
}

template class sqaod::CPUDenseGraphBFSolver<float>;
template class sqaod::CPUDenseGraphBFSolver<double>;
