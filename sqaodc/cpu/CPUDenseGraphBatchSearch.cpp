#include "CPUDenseGraphBatchSearch.h"
#include "CPUFormulas.h"
#include <float.h>

using namespace sqaod_cpu;
namespace sq = sqaod;

template<class real>
CPUDenseGraphBatchSearch<real>::CPUDenseGraphBatchSearch() {
}

template<class real>
void CPUDenseGraphBatchSearch<real>::setQUBO(const Matrix &W, sq::SizeType tileSize) {
    W_.map(W.data, W.rows, W.cols);
    tileSize_ = tileSize;
}

template<class real>
void CPUDenseGraphBatchSearch<real>::initSearch() {
    Emin_ = FLT_MAX;
    packedXList_.clear();
}


template<class real>
void CPUDenseGraphBatchSearch<real>::searchRange(sq::PackedBitSet xBegin, sq::PackedBitSet xEnd) {
    int nBatchSize = int(xEnd - xBegin);
    int N = W_.rows;

    Matrix bitsSeq(nBatchSize, N);
    Vector Ebatch;
    sq::createBitSetSequence(bitsSeq.data, N, xBegin, xEnd);
    DGFuncs<real>::calculate_E(&Ebatch, W_, bitsSeq);

    for (int idx = 0; idx < nBatchSize; ++idx) {
        real Etmp = Ebatch(idx);
        if (Etmp > Emin_) {
            continue;
        }
        else if (Etmp == Emin_) {
            if (packedXList_.size() < tileSize_)
                packedXList_.pushBack(xBegin + idx);
        }
        else {
            Emin_ = Etmp;
            packedXList_.clear();
            packedXList_.pushBack(xBegin + idx);
        }
    }
}

template struct sqaod_cpu::CPUDenseGraphBatchSearch<float>;
template struct sqaod_cpu::CPUDenseGraphBatchSearch<double>;
