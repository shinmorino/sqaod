#pragma once

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod_cpu {

template<class real>
struct CPUDenseGraphBatchSearch {
    typedef sqaod::MatrixType<real> Matrix;
    typedef sqaod::VectorType<real> Vector;
    
    CPUDenseGraphBatchSearch();

    void setProblem(const Matrix &W, sqaod::SizeType tileSize);

    void initSearch();
    
    void searchRange(sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);

    Matrix W_;
    sqaod::SizeType tileSize_;
    real Emin_;
    sqaod::PackedBitsArray packedXList_;
};


}
