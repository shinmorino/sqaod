#include "CPUBipartiteGraphBatchSearch.h"
#include <common/EigenBridge.h>
#include <cpu/CPUFormulas.h>
#include <float.h>

using namespace sqaod_cpu;
namespace sq = sqaod;

template<class real>
CPUBipartiteGraphBatchSearch<real>::CPUBipartiteGraphBatchSearch() {
}

template<class real> void CPUBipartiteGraphBatchSearch<real>::
setProblem(const Vector &b0, const Vector &b1, const Matrix &W,
           sqaod::SizeType tileSize0, sqaod::SizeType tileSize1) {
    b0_.map(b0.data, b0.size);
    b1_.map(b1.data, b1.size);
    W_.map(W.data, W.rows, W.cols);
    tileSize0_ = tileSize0;
    tileSize1_ = tileSize1;
}

template<class real>
void CPUBipartiteGraphBatchSearch<real>::initSearch() {
    Emin_ = FLT_MAX;
    packedXPairList_.clear();
}


template<class real> void CPUBipartiteGraphBatchSearch<real>::
searchRange(sq::PackedBits x0begin, sq::PackedBits x0end,
            sq::PackedBits x1begin, sq::PackedBits x1end) {
    int nBatchSize0 = int(x0end - x0begin);
    int nBatchSize1 = int(x1end - x1begin);

    // typedef sq::EigenColumnVectorType<real> EigenColumnVector;
    int N0 = W_.cols;
    int N1 = W_.rows;
    Matrix bitsSeq0(nBatchSize0, N0);
    Matrix bitsSeq1(nBatchSize1, N1);

    sq::createBitsSequence(bitsSeq0.data, N0, x0begin, x0end);
    sq::createBitsSequence(bitsSeq1.data, N1, x1begin, x1end);

    int maxNSolutions = W_.rows + W_.cols;
    Matrix EBatch;

    sq::BGFuncs<real>::calculate_E_2d(&EBatch, b0_, b1_, W_, bitsSeq0, bitsSeq1);
    for (int idx1 = 0; idx1 < nBatchSize1; ++idx1) {
        for (int idx0 = 0; idx0 < nBatchSize0; ++idx0) {
            real Etmp = EBatch(idx1, idx0);
            if (Etmp > Emin_) {
                continue;
            }
            else if (Etmp == Emin_) {
                if (packedXPairList_.size() < maxNSolutions)
                    packedXPairList_.pushBack(
                            sq::PackedBitsPairArray::ValueType(x0begin + idx0, x1begin + idx1));
            }
            else {
                Emin_ = Etmp;
                packedXPairList_.clear();
                packedXPairList_.pushBack(sq::PackedBitsPairArray::ValueType(x0begin + idx0, x1begin + idx1));
            }
        }
    }
}
    

template struct sqaod_cpu::CPUBipartiteGraphBatchSearch<float>;
template struct sqaod_cpu::CPUBipartiteGraphBatchSearch<double>;
