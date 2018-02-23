#pragma once

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod_cpu {

template<class real>
struct CPUBipartiteGraphBatchSearch {
    typedef sqaod::MatrixType<real> Matrix;
    typedef sqaod::VectorType<real> Vector;
    
    CPUBipartiteGraphBatchSearch();

    void setProblem(const Vector &b0, const Vector &b1, const Matrix &W,
                    sqaod::SizeType tileSize0, sqaod::SizeType tileSize1);

    void initSearch();
    
    void searchRange(sqaod::PackedBits x0begin, sqaod::PackedBits x0end,
                     sqaod::PackedBits x1begin, sqaod::PackedBits x1end);

    Vector b0_, b1_;
    Matrix W_;
    sqaod::SizeType tileSize0_;
    sqaod::SizeType tileSize1_;
    real Emin_;
    sqaod::PackedBitsPairArray packedXPairList_;
};


}
