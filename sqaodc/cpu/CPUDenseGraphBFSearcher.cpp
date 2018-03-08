#include "CPUDenseGraphBFSearcher.h"
#include <cpu/CPUDenseGraphBatchSearch.h>
#include <cmath>

#include <float.h>
#include <algorithm>

using namespace sqaod_cpu;

template<class real>
CPUDenseGraphBFSearcher<real>::CPUDenseGraphBFSearcher() {
    tileSize_ = 1024;
#ifdef _OPENMP
    nMaxThreads_ = omp_get_max_threads();
    sq::log("# max threads: %d", nMaxThreads_);
#else
    nMaxThreads_ = 1;
#endif
    searchers_ = new BatchSearcher[nMaxThreads_];
}

template<class real>
CPUDenseGraphBFSearcher<real>::~CPUDenseGraphBFSearcher() {
    delete [] searchers_;
    searchers_ = NULL;
}

template<class real>
void CPUDenseGraphBFSearcher<real>::setProblem(const Matrix &W, sq::OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    throwErrorIf(63 < N_, "N must be smaller than 64, N=%d.", N_);
    if (N_ != W.rows)
        clearState(solInitialized);

    N_ = W.rows;
    W_ = W;
    om_ = om;
    if (om_ == sq::optMaximize)
        W_ *= real(-1.);

    setState(solProblemSet);
}

template<class real>
sq::Preferences CPUDenseGraphBFSearcher<real>::getPreferences() const {
    sq::Preferences prefs = Base::getPreferences();
    prefs.pushBack(sq::Preference(sq::pnDevice, "cpu"));
    return prefs;
}

template<class real>
const sq::BitsArray &CPUDenseGraphBFSearcher<real>::get_x() const {
    throwErrorIfSolutionNotAvailable();
    return xList_;
}

template<class real>
const sq::VectorType<real> &CPUDenseGraphBFSearcher<real>::get_E() const {
    throwErrorIfSolutionNotAvailable();
    return E_;
}

template<class real>
void CPUDenseGraphBFSearcher<real>::initSearch() {
    Emin_ = FLT_MAX;
    xList_.clear();
    x_ = 0;
    xMax_ = 1ull << N_;
    if (xMax_ < tileSize_) {
        tileSize_ = (sq::SizeType)xMax_;
        sq::log("Tile size is adjusted to %d for N=%d", tileSize_, N_);
    }
    for (int idx = 0; idx < nMaxThreads_; ++idx) {
        searchers_[idx].setProblem(W_, tileSize_);
        searchers_[idx].initSearch();
    }
    setState(solInitialized);
}


template<class real>
void CPUDenseGraphBFSearcher<real>::finSearch() {
    xList_.clear();

    sq::PackedBitsArray packedXList;
    for (int idx = 0; idx < nMaxThreads_; ++idx) {
        const BatchSearcher &searcher = searchers_[idx];
        if (searcher.Emin_ < Emin_) {
            Emin_ = searcher.Emin_;
            packedXList = searcher.packedXList_;
        }
        else if (searcher.Emin_ == Emin_) {
            if (packedXList.size() < tileSize_) {
                packedXList.insert(searcher.packedXList_.begin(),
                                   searcher.packedXList_.end());
            }
        }
    }
    
    std::sort(packedXList.begin(), packedXList.end());
    int nSolutions = std::min(tileSize_, (sq::SizeType)packedXList.size());
    for (int idx = 0; idx < nSolutions; ++idx) {
        sq::Bits bits;
        unpackBits(&bits, packedXList[idx], N_);
        xList_.pushBack(bits);
    }
    E_.resize(nSolutions);
    mapToRowVector(E_).array() = (om_ == sq::optMaximize) ? - Emin_ : Emin_;

    setState(solSolutionAvailable);
}


template<class real>
bool CPUDenseGraphBFSearcher<real>::searchRange(sq::PackedBits *curXEnd) {
    throwErrorIfNotInitialized();
    
#ifdef _OPENMP
#pragma omp parallel
    {
        sq::SizeType threadNum = omp_get_thread_num();
        sq::PackedBits batchBegin = x_ + tileSize_ * threadNum;
        sq::PackedBits batchEnd = x_ + tileSize_ * (threadNum + 1);
        batchBegin = std::min(std::max(0ULL, batchBegin), xMax_);
        batchEnd = std::min(std::max(0ULL, batchEnd), xMax_);
        if (batchBegin < batchEnd)
            searchers_[threadNum].searchRange(batchBegin, batchEnd);
    }
    x_ = std::min(sq::PackedBits(x_ + tileSize_ * nMaxThreads_), xMax_);
#else
    sq::PackedBits batchBegin = x_;
    sq::PackedBits batchEnd = std::min(x_ + tileSize_, xMax_); ;
    if (batchBegin < batchEnd)
        searchers_[threadNum].searchRange(batchBegin, batchEnd);
    
    searchers_[0].searchRange(xBegin, xEnd);
    x_ = batchEnd;
#endif
    if (curXEnd != NULL)
        *curXEnd = x_;
    return (xMax_ == x_);
}

template class CPUDenseGraphBFSearcher<float>;
template class CPUDenseGraphBFSearcher<double>;
