#include "CPUBipartiteGraphBFSolver.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>

using namespace sqaod;

template<class real>
CPUBipartiteGraphBFSolver<real>::CPUBipartiteGraphBFSolver() {
    tileSize0_ = 1024;
    tileSize1_ = 1024;
}

template<class real>
CPUBipartiteGraphBFSolver<real>::~CPUBipartiteGraphBFSolver() {
}


template<class real>
void CPUBipartiteGraphBFSolver<real>::seed(unsigned long seed) {
    random_.seed(seed);
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::getProblemSize(int *N0, int *N1) const {
    *N0 = N0_;
    *N1 = N1_;
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::setProblem(const real *b0, const real *b1, const real *W,
                                                 int N0, int N1, OptimizeMethod om) {
    N0_ = N0;
    N1_ = N1;
    b0_ = Eigen::Map<RowVector>((real*)b0, N0_);
    b1_ = Eigen::Map<RowVector>((real*)b1, N1_);
    W_ = Eigen::Map<Matrix>((real*)W, N1_, N0_);
    om_ = om;
    if (om_ == optMaximize)
        W_ *= real(-1.);
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::setTileSize(int tileSize0, int tileSize1) {
    tileSize0_ = tileSize0;
    tileSize1_ = tileSize1;
}

template<class real>
const BitVectorPairArray &CPUBipartiteGraphBFSolver<real>::get_x() const {
    //unpackIntArrayToMatrix(bitX_, xList_, N_);
    return xPairs_;
}

template<class real>
real CPUBipartiteGraphBFSolver<real>::get_E() const {
    return (om_ == optMaximize) ? -E_ : E_;
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::initSearch() {
    E_ = FLT_MAX;
    xPackedParis_.clear();
    x0max_ = 1 << N0_;
    x1max_ = 1 << N1_;
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::searchRange(PackedBits iBegin0, PackedBits iEnd0,
                                                  PackedBits iBegin1, PackedBits iEnd1) {
    iBegin0 = std::min(std::max(0ULL, iBegin0), x0max_);
    iEnd0 = std::min(std::max(0ULL, iEnd0), x0max_);
    iBegin1 = std::min(std::max(0ULL, iBegin1), x1max_);
    iEnd1 = std::min(std::max(0ULL, iEnd1), x1max_);

    BGFuncs<real>::batchSearch(&E_, &xPackedParis_, b0_ ,b1_, W_, iBegin0, iEnd0, iBegin1, iEnd1);
    /* FIXME: add max limits of # min vectors. */
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::search() {
    int iStep0 = (int)std::min((unsigned long long)tileSize0_, x0max_);
    int iStep1 = (int)std::min((unsigned long long)tileSize1_, x1max_);

    initSearch();
    for (PackedBits iTile1 = 0; iTile1 < x1max_; iTile1 += iStep1) {
        for (PackedBits iTile0 = 0; iTile0 < x0max_; iTile0 += iStep0) {
            searchRange(iTile0, iTile0 + iStep0, iTile1, iTile1 + iStep1);
        }
    }
}

template class sqaod::CPUBipartiteGraphBFSolver<float>;
template class sqaod::CPUBipartiteGraphBFSolver<double>;
