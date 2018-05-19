#pragma once

#include <sqaodc/common/Matrix.h>
#include <sqaodc/common/Array.h>
#include <sqaodc/common/Random.h>
#include <sqaodc/common/Preference.h>
#include <sqaodc/common/Solver.h>
#include <sqaodc/common/Formulas.h>
#include <sqaodc/common/os_dependent.h>

namespace sqaod {

bool isCUDAAvailable();


template<class V>
void createBitSetSequence(V *bits, SizeType stride, SizeType nBits, PackedBitSet bBegin, PackedBitSet bEnd);
    
void unpackBitSet(BitSet *unpacked, const PackedBitSet packed, int N);
    

template<class real>
bool isSymmetric(const MatrixType<real> &W);

template<class V>
BitMatrix x_from_q(const MatrixType<V> &q);

template<class V>
BitSet x_from_q(const VectorType<V> &q);

template<class V>
MatrixType<V> x_to_q(const BitMatrix &x);

template<class V>
VectorType<V> x_to_q(const BitSet &x);

}
