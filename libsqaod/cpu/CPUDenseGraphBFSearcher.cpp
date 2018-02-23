#include "CPUDenseGraphBFSearcher.h"
#include <cpu/CPUDenseGraphBatchSearch.h>
#include <cmath>

#include <float.h>
#include <algorithm>

using namespace sqaod;

template<class real>
CPUDenseGraphBFSearcher<real>::CPUDenseGraphBFSearcher() {
    tileSize_ = 1024;
#ifdef _OPENMP
    nProcs_ = omp_get_num_procs();
    log("# processors: %d", nProcs_);
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
void CPUDenseGraphBFSearcher<real>::setProblem(const Matrix &W, OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    throwErrorIf(63 < N_, "N must be smaller than 64, N=%d.", N_);
    N_ = W.rows;
    W_ = W;
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
    Emin_ = FLT_MAX;
    xList_.clear();
    xMax_ = 1ull << N_;
    if (xMax_ < tileSize_) {
        tileSize_ = (SizeType)xMax_;
        log("Tile size is adjusted to %d for N=%d", tileSize_, N_);
    }
    for (int idx = 0; idx < nProcs_; ++idx)
        searchers_[idx].setProblem(W_, tileSize_);
}


template<class real>
void CPUDenseGraphBFSearcher<real>::finSearch() {
    xList_.clear();

    PackedBitsArray packedXList;
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
    int nSolutions = std::min(tileSize_, (SizeType)packedXList.size());
    for (int idx = 0; idx < nSolutions; ++idx) {
        Bits bits;
        unpackBits(&bits, packedXList[idx], N_);
        xList_.pushBack(bits);
    }
    E_.resize(nSolutions);
    mapToRowVector(E_).array() = (om_ == optMaximize) ? - Emin_ : Emin_;
}


template<class real>
void CPUDenseGraphBFSearcher<real>::searchRange(PackedBits xBegin, PackedBits xEnd) {
    xBegin = std::min(std::max(0ULL, xBegin), xMax_);
    xEnd = std::min(std::max(0ULL, xEnd), xMax_);
    
// #undef _OPENMP
#ifdef _OPENMP
    SizeType nBatchSize = (SizeType)(xEnd - xBegin);
#pragma omp parallel
    {
        SizeType threadNum = omp_get_thread_num();
        SizeType nBatchSizePerThread = (nBatchSize + nProcs_ - 1) / nProcs_;
        PackedBits batchBegin = xBegin + nBatchSizePerThread * threadNum;
        PackedBits batchEnd = xBegin + std::min(nBatchSize, nBatchSizePerThread * (threadNum + 1));
        searchers_[threadNum].searchRange(batchBegin, batchEnd);
    }
#else
    searchers_[0].searchRange(xBegin, xEnd);
#endif
}

template class sqaod::CPUDenseGraphBFSearcher<float>;
template class sqaod::CPUDenseGraphBFSearcher<double>;
