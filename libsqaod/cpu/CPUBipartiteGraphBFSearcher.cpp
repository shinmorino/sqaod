#include "CPUBipartiteGraphBFSearcher.h"
#include "CPUFormulas.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>

using namespace sqaod;

template<class real>
CPUBipartiteGraphBFSearcher<real>::CPUBipartiteGraphBFSearcher() {
    tileSize0_ = 1024;
    tileSize1_ = 1024;
}

template<class real>
CPUBipartiteGraphBFSearcher<real>::~CPUBipartiteGraphBFSearcher() {
}


template<class real>
void CPUBipartiteGraphBFSearcher<real>::setProblem(const Vector &b0, const Vector &b1,
                                                   const Matrix &W, OptimizeMethod om) {
    N0_ = b0.size;
    N1_ = b1.size;
    throwErrorIf(63 < N0_, "N0 must be smaller than 64, N0=%d.", N0_);
    throwErrorIf(63 < N1_, "N1 must be smaller than 64, N1=%d.", N1_);
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
const BitsPairArray &CPUBipartiteGraphBFSearcher<real>::get_x() const {
    return xPairs_;
}

template<class real>
const VectorType<real> &CPUBipartiteGraphBFSearcher<real>::get_E() const {
    return E_;
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::initSearch() {
    minE_ = FLT_MAX;
    xPackedPairs_.clear();
    x0max_ = 1ull << N0_;
    x1max_ = 1ull << N1_;
    if (x0max_ < tileSize0_) {
        tileSize0_ = (SizeType)x0max_;
        log("Tile size 0 is adjusted to %d for N0=%d", tileSize0_, N0_);
    }
    if (x1max_ < tileSize1_) {
        tileSize1_ = (SizeType)x1max_;
        log("Tile size 1 is adjusted to %d for N1=%d", tileSize1_, N1_);
    }
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::finSearch() {
    xPairs_.clear();
    for (PackedBitsPairArray::const_iterator it = xPackedPairs_.begin();
         it != xPackedPairs_.end(); ++it) {
        Bits x0(N0_), x1(N1_);
        unpackBits(&x0, it->bits0, N0_);
        unpackBits(&x1, it->bits1, N1_);
        xPairs_.pushBack(BitsPairArray::ValueType(x0, x1));
    }
    real tmpE = (om_ == optMaximize) ? -minE_ : minE_;
    E_.resize((SizeType)xPackedPairs_.size());
    mapToRowVector(E_).array() = tmpE;
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::searchRange(PackedBits iBegin0, PackedBits iEnd0,
                                                    PackedBits iBegin1, PackedBits iEnd1) {
    iBegin0 = std::min(std::max(0ULL, iBegin0), x0max_);
    iEnd0 = std::min(std::max(0ULL, iEnd0), x0max_);
    iBegin1 = std::min(std::max(0ULL, iBegin1), x1max_);
    iEnd1 = std::min(std::max(0ULL, iEnd1), x1max_);

    BGFuncs<real>::batchSearch(&minE_, &xPackedPairs_, b0_ ,b1_, W_, iBegin0, iEnd0, iBegin1, iEnd1);
    /* FIXME: add max limits of # min vectors. */
}

template class sqaod::CPUBipartiteGraphBFSearcher<float>;
template class sqaod::CPUBipartiteGraphBFSearcher<double>;
