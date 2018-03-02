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
    nProcs_ = omp_get_num_procs();
    sq::log("# processors: %d", nProcs_);
#else
    nProcs_ = 1;
#endif
    searchers_ = new BatchSearcher[nProcs_];
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
    xMax_ = 1ull << N_;
    if (xMax_ < tileSize_) {
        tileSize_ = (sq::SizeType)xMax_;
        sq::log("Tile size is adjusted to %d for N=%d", tileSize_, N_);
    }
    for (int idx = 0; idx < nProcs_; ++idx) {
        searchers_[idx].setProblem(W_, tileSize_);
        searchers_[idx].initSearch();
    }
    setState(solInitialized);
}


template<class real>
void CPUDenseGraphBFSearcher<real>::finSearch() {
    xList_.clear();

    sq::PackedBitsArray packedXList;
    for (int idx = 0; idx < nProcs_; ++idx) {
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
void CPUDenseGraphBFSearcher<real>::searchRange(sq::PackedBits xBegin, sq::PackedBits xEnd) {
    throwErrorIfNotInitialized();

    xBegin = std::min(std::max(0ULL, xBegin), xMax_);
    xEnd = std::min(std::max(0ULL, xEnd), xMax_);
    
// #undef _OPENMP
#ifdef _OPENMP
    sq::SizeType nBatchSize = (sq::SizeType)(xEnd - xBegin);
#pragma omp parallel
    {
        sq::SizeType threadNum = omp_get_thread_num();
        sq::SizeType nBatchSizePerThread = (nBatchSize + nProcs_ - 1) / nProcs_;
        sq::PackedBits batchBegin = xBegin + nBatchSizePerThread * threadNum;
        sq::PackedBits batchEnd = xBegin + std::min(nBatchSize, nBatchSizePerThread * (threadNum + 1));
        searchers_[threadNum].searchRange(batchBegin, batchEnd);
    }
#else
    searchers_[0].searchRange(xBegin, xEnd);
#endif
}

template class CPUDenseGraphBFSearcher<float>;
template class CPUDenseGraphBFSearcher<double>;
