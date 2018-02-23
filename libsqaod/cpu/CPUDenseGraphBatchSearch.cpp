#include "CPUDenseGraphBatchSearch.h"
#include <float.h>

using namespace sqaod_cpu;
namespace sq = sqaod;

template<class real>
CPUDenseGraphBatchSearch<real>::CPUDenseGraphBatchSearch() {
}

template<class real>
void CPUDenseGraphBatchSearch<real>::setProblem(const Matrix &W, sq::SizeType tileSize) {
    W_.map(W.data, W.rows, W.cols);
    tileSize_ = tileSize;
}

template<class real>
void CPUDenseGraphBatchSearch<real>::initSearch() {
    Emin_ = FLT_MAX;
    packedXList_.clear();
}


template<class real>
void CPUDenseGraphBatchSearch<real>::searchRange(sq::PackedBits xBegin, sq::PackedBits xEnd) {
    int nBatchSize = int(xEnd - xBegin);
    int N = W_.rows;

    EigenMappedMatrix eW(sq::mapTo(W_));
    
    EigenMatrix eBitsSeq(nBatchSize, N);
    sq::createBitsSequence(eBitsSeq.data(), N, xBegin, xEnd);

    EigenMatrix eWx = eW * eBitsSeq.transpose();
    EigenMatrix prod = eWx.transpose().cwiseProduct(eBitsSeq);
    sq::EigenColumnVectorType<real> eEbatch = prod.rowwise().sum(); 

    for (int idx = 0; idx < nBatchSize; ++idx) {
        if (eEbatch(idx) > Emin_) {
            continue;
        }
        else if (eEbatch(idx) == Emin_) {
            if (packedXList_.size() < tileSize_)
                packedXList_.pushBack(xBegin + idx);
        }
        else {
            Emin_ = eEbatch(idx);
            packedXList_.clear();
            packedXList_.pushBack(xBegin + idx);
        }
    }
}

template struct sqaod_cpu::CPUDenseGraphBatchSearch<float>;
template struct sqaod_cpu::CPUDenseGraphBatchSearch<double>;
