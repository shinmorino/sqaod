#pragma once

#include <sqaodc/common/Matrix.h>
#include <sqaodc/common/Array.h>
#include <sqaodc/common/Random.h>
#include <sqaodc/common/Preference.h>
#include <sqaodc/common/Solver.h>

namespace sqaod {


template<class real>
void createBitsSequence(real *bits, int nBits, PackedBits bBegin, PackedBits bEnd);
    
void unpackBits(Bits *unpacked, const PackedBits packed, int N);
    

template<class real>
bool isSymmetric(const MatrixType<real> &W);

template<class V> inline
BitMatrix x_from_q(const MatrixType<V> &q) {
    BitMatrix x(q.dim());
    x_from_q(x.data, q.data, q.rows * q.cols);
    return x;
}

template<class V> inline
Bits x_from_q(const VectorType<V> &q) {
    Bits x(q.size);
    x_from_q(x.data, q.data, x.size);
    return x;
}

template<class V> inline
MatrixType<V> x_to_q(const BitMatrix &x) {
    MatrixType<V> q(x.dim());
    x_to_q(q.data, x.data, x.rows * x.cols);
    return x;
}

template<class V> inline
VectorType<V> x_to_q(const Bits &x) {
    VectorType<V> q(x.size);
    x_from_q(q.data, x.data, x.size);
    return q;
}

}
