#pragma once

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod_cpu {

namespace sq = sqaod;

template<class real>
struct CPUDenseGraphBatchSearch {
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;
    
    CPUDenseGraphBatchSearch();

    void setQUBO(const Matrix &W, sq::SizeType tileSize);

    void initSearch();
    
    void searchRange(sq::PackedBitSet xBegin, sq::PackedBitSet xEnd);

    Matrix W_;
    sq::SizeType tileSize_;
    real Emin_;
    sq::PackedBitSetArray packedXList_;
};


}
