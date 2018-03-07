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
    if ((N0_!= b0.size) || (N1_ != b1.size))
        clearState(solInitialized);

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
    throwErrorIfSolutionNotAvailable();
    return xPairList_;
}

template<class real>
const sq::VectorType<real> &CPUBipartiteGraphBFSearcher<real>::get_E() const {
    throwErrorIfSolutionNotAvailable();
    return E_;
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::initSearch() {
    Emin_ = FLT_MAX;
    xPairList_.clear();
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
    setState(solInitialized);
}

template<class real>
void CPUBipartiteGraphBFSearcher<real>::finSearch() {
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
void CPUBipartiteGraphBFSearcher<real>::searchRange(sq::PackedBits x0begin, sq::PackedBits x0end,
                                                    sq::PackedBits x1begin, sq::PackedBits x1end) {
    throwErrorIfNotInitialized();
    
    x0begin = std::min(std::max(0ULL, x0begin), x0max_);
    x0end = std::min(std::max(0ULL, x0end), x0max_);
    x1begin = std::min(std::max(0ULL, x1begin), x1max_);
    x1end = std::min(std::max(0ULL, x1end), x1max_);
    
#ifdef _OPENMP
    sq::SizeType nBatchSize1 = (sq::SizeType)(x1end - x1begin);
#pragma omp parallel
    {
        sq::SizeType threadNum = omp_get_thread_num();
        sq::SizeType nBatchSize1PerThread = (nBatchSize1 + nMaxThreads_ - 1) / nMaxThreads_;
        sq::PackedBits batchBegin1 = x1begin + nBatchSize1PerThread * threadNum;
        sq::PackedBits batchEnd1 = x1begin +
                std::min(nBatchSize1, nBatchSize1PerThread * (threadNum + 1));
        searchers_[threadNum].searchRange(x0begin, x0end, batchBegin1, batchEnd1);
    }
#else
    searchers_[0].searchRange(x0begin, x0end, x1begin, x1end);
#endif
}

template class CPUBipartiteGraphBFSearcher<float>;
template class CPUBipartiteGraphBFSearcher<double>;
