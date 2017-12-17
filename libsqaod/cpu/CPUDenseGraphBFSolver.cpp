#include "CPUDenseGraphBFSolver.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>

using namespace sqaod;

template<class real>
CPUDenseGraphBFSolver<real>::CPUDenseGraphBFSolver() {
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
void CPUDenseGraphBFSolver<real>::setProblemSize(int N) {
    if (63 < N)
        throw std::exception(); /* FIXME: */
    N_ = N;
    W_.resize(N, N);
}


template<class real>
void CPUDenseGraphBFSolver<real>::setProblem(const real *W, OptimizeMethod om) {
    W_ = Eigen::Map<Matrix>((real*)W, N_, N_);
    om_ = om;
    if (om_ == optMaximize)
        W_ *= real(-1.);
}

template<class real>
const BitMatrix &CPUDenseGraphBFSolver<real>::get_x() const {
    unpackIntArrayToMatrix(bitX_, xList_, N_);
    return bitX_;
}

template<class real>
real CPUDenseGraphBFSolver<real>::get_E() const {
    return (om_ == optMaximize) ? -E_ : E_;
}

template<class real>
void CPUDenseGraphBFSolver<real>::initSearch() {
    E_ = FLT_MAX;
    xList_.clear();
    xMax_ = 1 << N_;
}

template<class real>
void CPUDenseGraphBFSolver<real>::searchRange(unsigned long long iBegin, unsigned long long iEnd) {
    iBegin = std::min(std::max(0ULL, iBegin), xMax_);
    iEnd = std::min(std::max(0ULL, iEnd), xMax_);
    DGFuncs<real>::batchSearch(&E_, &xList_, W_.data(), N_, iBegin, iEnd);
    /* FIXME: add max limits of # min vectors. */
}

template<class real>
void CPUDenseGraphBFSolver<real>::search() {
    const int nStep = 1024;
    int iStep = (int)std::min((unsigned long long)nStep, xMax_);

    initSearch();
    for (unsigned long long iTile = 0; iTile < xMax_; iTile += iStep) {
        searchRange(iTile, iTile + iStep);
    }
}

template class sqaod::CPUDenseGraphBFSolver<float>;
template class sqaod::CPUDenseGraphBFSolver<double>;
