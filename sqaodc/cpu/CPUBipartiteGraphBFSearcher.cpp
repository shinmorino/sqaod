#include "CPUBipartiteGraphBFSearcher.h"
#include "CPUBipartiteGraphBatchSearch.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>

using namespace sqaod_cpu;

template<class real>
CPUBipartiteGraphBFSearcher<real>::CPUBipartiteGraphBFSearcher() {
    tileSize0_ = 1024;
    tileSize1_ = 1024;
#ifdef _OPENMP
    nMaxThreads_ = omp_get_max_threads();
    sq::log("# max threads: %d", nMaxThreads_);
#else
    nMaxThreads_ = 1;
#endif
    searchers_ = new BatchSearcher[nMaxThreads_];
}

template<class real>
CPUBipartiteGraphBFSearcher<real>::~CPUBipartiteGraphBFSearcher() {
    delete [] searchers_;
    searchers_ = NULL;
}


template<class real>
void CPUBipartiteGraphBFSearcher<real>::setProblem(const Vector &b0, const Vector &b1,
                                                   const Matrix &W, sq::OptimizeMethod om) {
    clearState(solProblemSet);

    N0_ = b0.size;
    N1_ = b1.size;
    throwErrorIf(63 < N0_, "N0 must be smaller than 64, N0=%d.", N0_);
    throwErrorIf(63 < N1_, "N1 must be smaller than 64, N1=%d.", N1_);
    b0_ = b0;
    b1_ = b1;
    W_ = W;
    om_ = om;
    if (om_ == sq::optMaximize) {
        W_ *= real(-1.);
        b0_ *= real(-1.);
        b1_ *= real(-1.);
    }

    setState(solProblemSet);
}

template<class real>
sq::Preferences CPUBipartiteGraphBFSearcher<real>::getPreferences() const {
    sq::Preferences prefs = Base::getPreferences();
    prefs.pushBack(sq::Preference(sq::pnDevice, "cpu"));
    return prefs;
}

template<class real>
const sq::BitsPairArray &CPUBipartiteGraphBFSearcher<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return xPairList_;
}

template<class real>
const sq::VectorType<real> &CPUBipartiteGraphBFSearcher<real>::get_E() const {
    if (!isEAvailable())
        const_cast<This*>(this)->calculate_E();
    return E_;
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::prepare() {
    Emin_ = FLT_MAX;
    xPairList_.clear();
    x0_ = x1_ = 0;
    x0max_ = 1ull << N0_;
    x1max_ = 1ull << N1_;
    if (x0max_ < tileSize0_) {
        tileSize0_ = (sq::SizeType)x0max_;
        sq::log("Tile size 0 is adjusted to %d for N0=%d", tileSize0_, N0_);
    }
    if (x1max_ < tileSize1_) {
        tileSize1_ = (sq::SizeType)x1max_;
        sq::log("Tile size 1 is adjusted to %d for N1=%d", tileSize1_, N1_);
    }
    for (int idx = 0; idx < nMaxThreads_; ++idx) {
        searchers_[idx].setProblem(b0_, b1_, W_, tileSize0_, tileSize1_);
        searchers_[idx].initSearch();
    }
    setState(solPrepared);
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::calculate_E() {
    throwErrorIfNotPrepared();
    if (xPairList_.empty())
        E_.resize(1);
    else
        E_.resize(xPairList_.size());
    mapToRowVector(E_).array() = (om_ == sq::optMaximize) ? - Emin_ : Emin_;
    setState(solEAvailable);
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::makeSolution() {
    xPairList_.clear();

    int nMaxSolutions = tileSize0_ + tileSize1_;
    
    sq::PackedBitsPairArray packedXPairList;
    for (int idx = 0; idx < nMaxThreads_; ++idx) {
        const BatchSearcher &searcher = searchers_[idx];
        if (searcher.Emin_ < Emin_) {
            Emin_ = searcher.Emin_;
            packedXPairList = searcher.packedXPairList_;
        }
        else if (searcher.Emin_ == Emin_) {
            if (packedXPairList.size() < nMaxSolutions) {
                packedXPairList.insert(searcher.packedXPairList_.begin(),
                                       searcher.packedXPairList_.end());
                                       
            }
        }
    }

    int nSolutions = std::min(nMaxSolutions, (int)packedXPairList.size());
    for (int idx = 0; idx < nSolutions; ++idx) {
        const sq::PackedBitsPair &pair =  packedXPairList[idx];
        sq::Bits x0(N0_), x1(N1_);
        unpackBits(&x0, pair.bits0, N0_);
        unpackBits(&x1, pair.bits1, N1_);
        xPairList_.pushBack(sq::BitsPairArray::ValueType(x0, x1));
    }
    real tmpE = (om_ == sq::optMaximize) ? - Emin_ : Emin_;
    E_.resize(nSolutions);
    mapToRowVector(E_).array() = tmpE;

    setState(solSolutionAvailable);
}

template<class real>
bool CPUBipartiteGraphBFSearcher<real>::searchRange(sq::PackedBits *curX0, sq::PackedBits *curX1) {
    throwErrorIfNotPrepared();
    clearState(solSolutionAvailable);
    
#ifdef _OPENMP

    sq::PackedBitsArray batch0begin(nMaxThreads_), batch0end(nMaxThreads_);
    sq::PackedBitsArray batch1begin(nMaxThreads_), batch1end(nMaxThreads_);

    /* calculate begin/end */
    sq::PackedBits x0 = x0_;
    sq::PackedBits x0end = std::min(x0 + tileSize0_, x0max_);
    sq::PackedBits x1 = x1_;
    for (int idx = 0; idx < nMaxThreads_; ++idx) {

        batch1begin.pushBack(x1);
        sq::PackedBits x1end = std::min(x1 + tileSize1_, x1max_);
        batch1end.pushBack(x1end);
        batch0begin.pushBack(x0);
        batch0end.pushBack(x0end);

        x1 = x1end;
        if (x1 == x1max_) {
            x1 = 0;
            /* move to the next x0 range. */
            x0 = std::min(x0 + tileSize0_, x0max_);
            x0end = std::min(x0 + tileSize0_, x0max_);
        }
    }
    
#pragma omp parallel
    {
        sq::SizeType threadNum = omp_get_thread_num();
        sq::PackedBits b0b = batch0begin[threadNum];
        sq::PackedBits b0e = batch0end[threadNum];
        sq::PackedBits b1b = batch1begin[threadNum];
        sq::PackedBits b1e = batch1end[threadNum];
        if ((b0b < b0e) && (b1b < b1e))
            searchers_[threadNum].searchRange(b0b, b0e, b1b, b1e);
    }

    /* move to next batch */
    x0_ = batch0begin[nMaxThreads_ - 1];
    x1_ = batch1end[nMaxThreads_ - 1];

#else
    sq::PackedBits batch0begin = x0_;
    sq::PackedBits batch0end = std::min(x0_ + tileSize0_, x0max_);
    sq::PackedBits batch1begin = x1_;
    sq::PackedBits batch1end = std::min(x1_ + tileSize1_, x1max_);
    if ((batch0begin < batch0end) && (batch1begin < batch1end))
        searchers_[0].searchRange(batch0begin, batch0end, batch1begin, batch1end);

    x1_ = batch1end;
#endif

    if (x1_ == x1max_) {
        x1_ = 0;
        x0_ = std::min(x0_ + tileSize0_, x0max_);
    }
    
    if (curX0 != NULL)
        *curX0 = x0_;
    if (curX1 != NULL)
        *curX1 = x1_;

    return (x0_ == x0max_);
}


template class CPUBipartiteGraphBFSearcher<float>;
template class CPUBipartiteGraphBFSearcher<double>;
