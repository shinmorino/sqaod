#include "CPUBipartiteGraphBFSolver.h"
#include "CPUFormulas.h"
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
void CPUBipartiteGraphBFSolver<real>::getProblemSize(int *N0, int *N1) const {
    *N0 = N0_;
    *N1 = N1_;
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::setProblem(const Vector &b0, const Vector &b1,
                                                 const Matrix &W, OptimizeMethod om) {
    N0_ = b0.size;
    N1_ = b1.size;
    b0_ = mapToRowVector(b0);
    b1_ = mapToRowVector(b1);
    W_ = mapTo(W);
    om_ = om;
    if (om_ == optMaximize) {
        W_ *= real(-1.);
        b0_ *= real(-1.);
        b1_ *= real(-1.);
    }
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::setTileSize(SizeType tileSize0, SizeType tileSize1) {
    tileSize0_ = tileSize0;
    tileSize1_ = tileSize1;
}

template<class real>
const BitsPairArray &CPUBipartiteGraphBFSolver<real>::get_x() const {
    return xPairs_;
}

template<class real>
const VectorType<real> &CPUBipartiteGraphBFSolver<real>::get_E() const {
    return E_;
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::initSearch() {
    minE_ = FLT_MAX;
    xPackedPairs_.clear();
    x0max_ = 1ull << N0_;
    x1max_ = 1ull << N1_;
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::finSearch() {
    xPairs_.clear();
    for (PackedBitsPairArray::const_iterator it = xPackedPairs_.begin();
         it != xPackedPairs_.end(); ++it) {
        Bits x0(N0_), x1(N1_);
        unpackBits(&x0, it->first, N0_);
        unpackBits(&x1, it->second, N1_);
        xPairs_.pushBack(BitsPairArray::ValueType(x0, x1));
    }
    real tmpE = (om_ == optMaximize) ? -minE_ : minE_;
    E_.resize((SizeType)xPackedPairs_.size());
    mapToRowVector(E_).array() = tmpE;
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::searchRange(PackedBits iBegin0, PackedBits iEnd0,
                                                  PackedBits iBegin1, PackedBits iEnd1) {
    iBegin0 = std::min(std::max(0ULL, iBegin0), x0max_);
    iEnd0 = std::min(std::max(0ULL, iEnd0), x0max_);
    iBegin1 = std::min(std::max(0ULL, iBegin1), x1max_);
    iEnd1 = std::min(std::max(0ULL, iEnd1), x1max_);

    BGFuncs<real>::batchSearch(&minE_, &xPackedPairs_, b0_ ,b1_, W_, iBegin0, iEnd0, iBegin1, iEnd1);
    /* FIXME: add max limits of # min vectors. */
}

template<class real>
void CPUBipartiteGraphBFSolver<real>::search() {
    PackedBits iStep0 = std::min(tileSize0_, x0max_);
    PackedBits iStep1 = std::min(tileSize1_, x1max_);

    initSearch();
    for (PackedBits iTile1 = 0; iTile1 < x1max_; iTile1 += iStep1) {
        for (PackedBits iTile0 = 0; iTile0 < x0max_; iTile0 += iStep0) {
            searchRange(iTile0, iTile0 + iStep0, iTile1, iTile1 + iStep1);
        }
    }
    finSearch();
}

template class sqaod::CPUBipartiteGraphBFSolver<float>;
template class sqaod::CPUBipartiteGraphBFSolver<double>;
