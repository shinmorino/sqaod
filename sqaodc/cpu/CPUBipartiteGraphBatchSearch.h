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
    
    void searchRange(sq::PackedBits x0begin, sq::PackedBits x0end,
                     sq::PackedBits x1begin, sq::PackedBits x1end);

    Vector b0_, b1_;
    Matrix W_;
    sq::SizeType tileSize0_;
    sq::SizeType tileSize1_;
    real Emin_;
    sq::PackedBitsPairArray packedXPairList_;
};


}
