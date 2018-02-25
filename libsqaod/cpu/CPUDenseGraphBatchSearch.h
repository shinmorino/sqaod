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

    void setProblem(const Matrix &W, sq::SizeType tileSize);

    void initSearch();
    
    void searchRange(sq::PackedBits xBegin, sq::PackedBits xEnd);

    Matrix W_;
    sq::SizeType tileSize_;
    real Emin_;
    sq::PackedBitsArray packedXList_;
};


}
