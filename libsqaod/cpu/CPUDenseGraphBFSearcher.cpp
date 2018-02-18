#include "CPUDenseGraphBFSearcher.h"
#include "CPUFormulas.h"
#include <cmath>

#include <float.h>
#include <algorithm>

using namespace sqaod;

template<class real>
CPUDenseGraphBFSearcher<real>::CPUDenseGraphBFSearcher() {
    tileSize_ = 1024;
}

template<class real>
CPUDenseGraphBFSearcher<real>::~CPUDenseGraphBFSearcher() {
}

template<class real>
void CPUDenseGraphBFSearcher<real>::setProblem(const Matrix &W, OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    N_ = W.rows;
    W_ = mapTo(W);
    om_ = om;
    if (om_ == optMaximize)
        W_ *= real(-1.);
}

template<class real>
const BitsArray &CPUDenseGraphBFSearcher<real>::get_x() const {
    return xList_;
}

template<class real>
const VectorType<real> &CPUDenseGraphBFSearcher<real>::get_E() const {
    return E_;
}

template<class real>
void CPUDenseGraphBFSearcher<real>::initSearch() {
    minE_ = FLT_MAX;
    packedXList_.clear();
    xList_.clear();
    xMax_ = 1ull << N_;
}


template<class real>
void CPUDenseGraphBFSearcher<real>::finSearch() {
    xList_.clear();
    for (int idx = 0; idx < (int)packedXList_.size(); ++idx) {
        Bits bits;
        unpackBits(&bits, packedXList_[idx], N_);
        xList_.pushBack(bits);
    }
    E_.resize((SizeType)packedXList_.size());
    mapToRowVector(E_).array() = (om_ == optMaximize) ? - minE_ : minE_;
}


template<class real>
void CPUDenseGraphBFSearcher<real>::searchRange(unsigned long long iBegin, unsigned long long iEnd) {
    iBegin = std::min(std::max(0ULL, iBegin), xMax_);
    iEnd = std::min(std::max(0ULL, iEnd), xMax_);
    DGFuncs<real>::batchSearch(&minE_, &packedXList_, mapFrom(W_), iBegin, iEnd);
    /* FIXME: add max limits of # min vectors. */
}

template class sqaod::CPUDenseGraphBFSearcher<float>;
template class sqaod::CPUDenseGraphBFSearcher<double>;
