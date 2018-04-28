#pragma once

#include <sqaodc/common/Matrix.h>
#include <sqaodc/common/Array.h>
#include <sqaodc/common/Random.h>
#include <sqaodc/common/Preference.h>
#include <sqaodc/common/Solver.h>

namespace sqaod {

bool isCUDAAvailable();


template<class V>
void createBitSetSequence(V *bits, SizeType stride, SizeType nBits, PackedBitSet bBegin, PackedBitSet bEnd);
    
void unpackBitSet(BitSet *unpacked, const PackedBitSet packed, int N);
    

template<class real>
bool isSymmetric(const MatrixType<real> &W);

template<class V> inline
BitMatrix x_from_q(const MatrixType<V> &q) {
    BitMatrix x(q.dim());
    x_from_q(x.data, q.data, q.rows * q.cols);
    return x;
}

template<class V> inline
BitSet x_from_q(const VectorType<V> &q) {
    BitSet x(q.size);
    x_from_q(x.data, q.data, x.size);
    return x;
}

template<class V> inline
MatrixType<V> x_to_q(const BitMatrix &x) {
    MatrixType<V> q(x.dim());
    x_to_q(q.data, x.data, x.cols, x.rows, x.stride);
    return x;
}

template<class V> inline
VectorType<V> x_to_q(const BitSet &x) {
    VectorType<V> q(x.size);
    x_to_q(q.data, x.data, x.size);
    return q;
}

}
