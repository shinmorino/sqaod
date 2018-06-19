#include "CPUDenseGraphBFSearcher.h"
#include "CPUDenseGraphBatchSearch.h"
#include <sqaodc/common/internal/ShapeChecker.h>
#include <cmath>

#include <float.h>
#include <algorithm>

namespace sqint = sqaod_internal;
using namespace sqaod_cpu;

template<class real>
CPUDenseGraphBFSearcher<real>::CPUDenseGraphBFSearcher() {
    tileSize_ = 1024;
    nWorkers_ = sq::getDefaultNumThreads();
    sq::log("# wokers: %d", nWorkers_);
    searchers_ = new BatchSearcher[nWorkers_];
}

template<class real>
CPUDenseGraphBFSearcher<real>::~CPUDenseGraphBFSearcher() {
    delete [] searchers_;
    searchers_ = NULL;
}

template<class real>
void CPUDenseGraphBFSearcher<real>::setQUBO(const Matrix &W, sq::OptimizeMethod om) {
    sqint::matrixCheckIfSymmetric(W, __func__);
    throwErrorIf(63 < W.rows, "N must be smaller than 64, N=%d.", W.rows);
    clearState(solProblemSet);

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
const sq::BitSetArray &CPUDenseGraphBFSearcher<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return xList_;
}

template<class real>
const sq::VectorType<real> &CPUDenseGraphBFSearcher<real>::get_E() const {
    if (!isEAvailable())
        const_cast<This*>(this)->calculate_E();
    return E_;
}

template<class real>
void CPUDenseGraphBFSearcher<real>::prepare() {
    Emin_ = FLT_MAX;
    xList_.clear();
    x_ = 0;
    xMax_ = 1ull << N_;
    if (xMax_ < (sq::PackedBitSet)tileSize_) {
        tileSize_ = sq::SizeType(xMax_);
        sq::log("Tile size is adjusted to %d for N=%d", tileSize_, N_);
    }
    for (int idx = 0; idx < nWorkers_; ++idx) {
        searchers_[idx].setQUBO(W_, tileSize_);
        searchers_[idx].initSearch();
    }

    if (nWorkers_ == 1)
        searchMethod_ = &CPUDenseGraphBFSearcher<real>::searchRangeSingleThread;
    else if (experiment_ == 0)
        searchMethod_ = &CPUDenseGraphBFSearcher<real>::searchRangeParallel;
    else
        searchMethod_ = &CPUDenseGraphBFSearcher<real>::searchRangeParallel2;

    setState(solPrepared);

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
    rangeMap_.clear();
#endif
}


template<class real>
void CPUDenseGraphBFSearcher<real>::calculate_E() {
    throwErrorIfNotPrepared();
    if (xList_.empty())
        E_.resize(1);
    else
        E_.resize(xList_.size());
    mapToRowVector(E_).array() = (om_ == sq::optMaximize) ? - Emin_ : Emin_;
    setState(solEAvailable);
}

template<class real>
void CPUDenseGraphBFSearcher<real>::makeSolution() {
    throwErrorIfNotPrepared();

    xList_.clear();
    sq::PackedBitSetArray packedXList;
    for (int idx = 0; idx < nWorkers_; ++idx) {
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
    int nSolutions = std::min(tileSize_, packedXList.size());
    for (int idx = 0; idx < nSolutions; ++idx) {
        sq::BitSet bits;
        sq::unpackBitSet(&bits, packedXList[idx], N_);
        xList_.pushBack(bits);
    }
    calculate_E();
    setState(solSolutionAvailable);


#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
    assert(rangeMap_.size() == 1);
    sq::PackedBitSetPair pair = rangeMap_[0];
    assert((pair.bits0 == 0) && (pair.bits1 == xMax_));
#endif
}


template<class real>
bool CPUDenseGraphBFSearcher<real>::searchRangeSingleThread(sq::PackedBitSet *curXEnd) {
    throwErrorIfNotPrepared();
    clearState(solSolutionAvailable);

    sq::PackedBitSet batchBegin = x_;
    sq::PackedBitSet batchEnd = std::min(x_ + tileSize_, xMax_); ;

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
    if (batchBegin < batchEnd)
        rangeMap_.insert(batchBegin, batchEnd);
#endif

    if (batchBegin < batchEnd)
        searchers_[0].searchRange(batchBegin, batchEnd);

    x_ = batchEnd;
    if (curXEnd != NULL)
        *curXEnd = x_;
    return (xMax_ == x_);
}


template<class real>
bool CPUDenseGraphBFSearcher<real>::searchRangeParallel(sq::PackedBitSet *curXEnd) {
    throwErrorIfNotPrepared();
    clearState(solSolutionAvailable);

#ifdef _OPENMP
#pragma omp parallel
    {
        sq::SizeType threadNum = omp_get_thread_num();
        sq::PackedBitSet batchBegin = x_ + tileSize_ * threadNum;
        sq::PackedBitSet batchEnd = x_ + tileSize_ * (threadNum + 1);
        batchBegin = std::min(std::max(0ULL, batchBegin), xMax_);
        batchEnd = std::min(std::max(0ULL, batchEnd), xMax_);

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
#pragma omp critical
        {
            rangeMap_.insert(batchBegin, batchEnd);
        }
#endif

        if (batchBegin < batchEnd)
            searchers_[threadNum].searchRange(batchBegin, batchEnd);
    }
    x_ = std::min(sq::PackedBitSet(x_ + tileSize_ * nWorkers_), xMax_);
    if (curXEnd != NULL)
        *curXEnd = x_;
    return (xMax_ == x_);

#else
    abort_("Should not reach here.");
    return false;
#endif
}


template<class real>
bool CPUDenseGraphBFSearcher<real>::searchRangeParallel2(sq::PackedBitSet *curXEnd) {
    throwErrorIfNotPrepared();
    clearState(solSolutionAvailable);

    auto searchWorker = [=](int threadIdx) {
        sq::PackedBitSet batchBegin = x_ + tileSize_ * threadIdx;
        sq::PackedBitSet batchEnd = x_ + tileSize_ * (threadIdx + 1);
        batchBegin = std::min(std::max(0ULL, batchBegin), xMax_);
        batchEnd = std::min(std::max(0ULL, batchEnd), xMax_);

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
        {
            std::unique_lock<std::mutex> lock(mutex_);
            rangeMap_.insert(batchBegin, batchEnd);
        }
#endif
        if (batchBegin < batchEnd)
            searchers_[threadIdx].searchRange(batchBegin, batchEnd);
    };
    parallel_.run(searchWorker);
    
    x_ = std::min(sq::PackedBitSet(x_ + tileSize_ * nWorkers_), xMax_);
    if (curXEnd != NULL)
        *curXEnd = x_;
    return (xMax_ == x_);
}



template class CPUDenseGraphBFSearcher<float>;
template class CPUDenseGraphBFSearcher<double>;
