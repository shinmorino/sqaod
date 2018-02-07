#ifndef COMMON_H__
#define COMMON_H__

#include <common/Matrix.h>
#include <common/Array.h>

namespace sqaod {

enum Algorithm {
    algoDefault,
    algoNaive,
    algoColored,
};

const char *algoToName(Algorithm algo);


enum OptimizeMethod {
    optMinimize,
    optMaximize
};

enum AnnealerState {
    annNone = 0,
    annRandSeedGiven = 1,
    annNTrottersGiven = 2,
    annQSet = 4,
};

template<class real>
void createBitsSequence(real *bits, int nBits, PackedBits bBegin, PackedBits bEnd);
    
void unpackBits(Bits *unpacked, const PackedBits packed, int N);
    

template<class real>
bool isSymmetric(const MatrixType<real> &W);

template<class V> inline
MatrixType<char> x_from_q(const MatrixType<V> &q) {
    MatrixType<char> x(q.size);
    x_from_q(x.data, q.data, q.rows * q.cols);
    return x;
}

template<class V> inline
VectorType<char> x_from_q(const VectorType<V> &q) {
    VectorType<char> x(q.size);
    x_from_q(x.data, q.data, x.size);
    return x;
}


}

#endif
