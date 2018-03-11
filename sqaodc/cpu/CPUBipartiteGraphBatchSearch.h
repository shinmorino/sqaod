#pragma once

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod_cpu {

namespace sq = sqaod;

template<class real>
struct CPUBipartiteGraphBatchSearch {
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;
    
    CPUBipartiteGraphBatchSearch();

    void setProblem(const Vector &b0, const Vector &b1, const Matrix &W,
                    sq::SizeType tileSize0, sq::SizeType tileSize1);

    void initSearch();
    
    void searchRange(sq::PackedBitSet x0begin, sq::PackedBitSet x0end,
                     sq::PackedBitSet x1begin, sq::PackedBitSet x1end);

    Vector b0_, b1_;
    Matrix W_;
    sq::SizeType tileSize0_;
    sq::SizeType tileSize1_;
    real Emin_;
    sq::PackedBitSetPairArray packedXPairList_;
};


}
