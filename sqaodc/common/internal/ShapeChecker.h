#pragma once

#include <sqaodc/common/Common.h>


namespace sqaod_internal {

namespace sq = sqaod;


/* Dense graph */

template<class real>
void matrixCheckIfSymmetric(const sq::MatrixType<real> &W, const char *func) {
    throwErrorIf(!sq::isSymmetric(W), "%s, Matrix is not symmetric.", func);
}

template<class real>
void quboShapeCheck(const sq::MatrixType<real> &W,
                    const char *func) {
    throwErrorIf(W.rows != W.cols, "%s, W is not a sqare matrix.", func);
}

template<class real>
void quboShapeCheck(const sq::MatrixType<real> &W,
                    const sq::VectorType<real> &x,
                    const char *func) {
    quboShapeCheck(W, func);
    throwErrorIf(W.cols != x.size, "%s, Shape does not match.", func);
}

template<class real>
void quboShapeCheck(const sq::MatrixType<real> &W,
                    const sq::MatrixType<real> &x,
                    const char *func) {
    throwErrorIf(W.rows != W.cols, "%s, W is not a sqare matrix.", func);
    throwErrorIf(W.cols != x.cols, "%s, Shape does not match.", func);
}

template<class real>
void isingModelShapeCheck(const sq::VectorType<real> &h,
                          const sq::MatrixType<real> &J, real c,
                          const char *func) {
    throwErrorIf(J.rows != J.cols, "%s, J is not a sqare matrix.", func);
    sq::SizeType N = J.cols;
    throwErrorIf(h.size != N, "%s, Shape does not match.", func);
}

template<class V0, class V1>
void isingModelShapeCheck(const sq::VectorType<V0> &h,
                          const sq::MatrixType<V0> &J, V0 c,
                          const sq::VectorType<V1> &q,
                          const char *func) {
    isingModelShapeCheck(h, J, c, func);
    sq::SizeType N = J.cols;
    throwErrorIf(q.size != N, "%s, Shape does not match.", func);
}

template<class V0, class V1>
void isingModelShapeCheck(const sq::VectorType<V0> &h,
                          const sq::MatrixType<V0> &J, V0 c,
                          const sq::MatrixType<V1> &q,
                          const char *func) {
    isingModelShapeCheck(h, J, c, func);
    sq::SizeType N = J.cols;
    throwErrorIf(q.cols != N, "%s, Shape does not match.", func);
}

template<class V0>
void isingModelShapeCheck(const sq::VectorType<V0> &h,
                          const sq::MatrixType<V0> &J, V0 c,
                          const sq::BitSetArray &q,
                          const char *func) {
    isingModelShapeCheck(h, J, c, func);
    sq::SizeType N = J.cols;
    for (int idx = 0; idx < q.size(); ++idx)
        throwErrorIf(q[idx].size != N, "%s, Shape does not match.", func);
}

template<class V>
void isingModelSolutionShapeCheck(sq::SizeType N,
                                  const sq::VectorType<V> &q,
                                  const char *func) {
    throwErrorIf(q.size != N, "%s, Shape does not match.", func);
}

inline
void isingModelSolutionShapeCheck(sq::SizeType N,
                                  const sq::BitSetArray &q,
                                  const char *func) {
    for (int idx = 0; idx < q.size(); ++idx)
        throwErrorIf(q[idx].size != N, "%s, Shape does not match.", func);
}

/* Bipartite graph */

template<class real>
void quboShapeCheck(const sq::VectorType<real> &b0,
                    const sq::VectorType<real> &b1,
                    const sq::MatrixType<real> &W,
                    const char *func) {
    bool shapeMatched = (W.cols == b0.size) && (W.rows == b1.size);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

template<class real>
void quboShapeCheck(const sq::VectorType<real> &b0,
                    const sq::VectorType<real> &b1,
                    const sq::MatrixType<real> &W,
                    const sq::VectorType<real> &x0,
                    const sq::VectorType<real> &x1,
                    const char *func) {
    sq::SizeType N0 = b0.size;
    sq::SizeType N1 = b1.size;
    bool shapeMatched = (x0.size == N0) && (x1.size == N1) &&
            (W.cols == N0) && (W.rows == N1);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

template<class real>
void quboShapeCheck(const sq::VectorType<real> &b0,
                    const sq::VectorType<real> &b1,
                    const sq::MatrixType<real> &W,
                    const sq::MatrixType<real> &x0,
                    const sq::MatrixType<real> &x1,
                    const char *func) {
    bool shapeMatched = (b0.size == x0.cols) && (b1.size == x1.cols) &&
            (W.cols == x0.cols) && (W.rows == x1.cols);
    shapeMatched &= (x0.rows == x1.rows);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

template<class real>
void quboShapeCheck_2d(const sq::VectorType<real> &b0,
                       const sq::VectorType<real> &b1,
                       const sq::MatrixType<real> &W,
                       const sq::MatrixType<real> &x0,
                       const sq::MatrixType<real> &x1,
                       const char *func) {
    bool shapeMatched = (b0.size == x0.cols) && (b1.size == x1.cols) &&
            (W.cols == x0.cols) && (W.rows == x1.cols);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

template<class real>
void isingModelShapeCheck(const sq::VectorType<real> &h0,
                          const sq::VectorType<real> &h1,
                          const sq::MatrixType<real> &J, real c,
                          const char *func) {
    bool shapeMatched = (h0.size == J.cols) && (h1.size == J.rows);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

template<class V0, class V1>
void isingModelShapeCheck(const sq::VectorType<V0> &h0,
                          const sq::VectorType<V0> &h1,
                          const sq::MatrixType<V0> &J, V0 c,
                          const sq::VectorType<V1> &q0,
                          const sq::VectorType<V1> &q1,
                          const char *func) {
    bool shapeMatched = (h0.size == q0.size) && (h1.size == q1.size) &&
            (J.cols == q0.size) && (J.rows == q1.size);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

template<class real>
void isingModelShapeCheck(const sq::VectorType<real> &h0,
                          const sq::VectorType<real> &h1,
                          const sq::MatrixType<real> &J, real c,
                          const sq::MatrixType<real> &q0,
                          const sq::MatrixType<real> &q1,
                          const char *func) {
    bool shapeMatched = (h0.size == q0.cols) && (h1.size == q1.cols) &&
            (J.cols == q0.cols) && (J.rows == q1.cols);
    shapeMatched &= (q0.rows == q1.rows);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

template<class V>
void isingModelSolutionShapeCheck(sq::SizeType N0, sq::SizeType N1,
                                  const sq::VectorType<V> &q0,
                                  const sq::VectorType<V> &q1,
                                  const char *func) {
    bool shapeMatched = (N0 == q0.size) && (N1 == q1.size);
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}

inline
void isingModelSolutionShapeCheck(sq::SizeType N0, sq::SizeType N1,
                                  const sq::BitSetPairArray &qPairs,
                                  const char *func) {
    bool shapeMatched = true;
    for (int idx = 0; idx < qPairs.size(); ++idx) {
        const sq::BitSet &q0 = qPairs[idx].bits0, &q1 = qPairs[idx].bits1;
        shapeMatched &= (N0 == q0.size) && (N1 == q1.size);
    }
    throwErrorIf(!shapeMatched, "%s, Shape does not match.", func);
}


/* helpers to check  out parameters */

template<class real> inline
void prepMatrix(sq::MatrixType<real> *mat, const sq::Dim &dim, const char *func) {
    throwErrorIf(mat == NULL, "%s, matrix is NULL.", func);
    if (mat->data == NULL)
        mat->resize(dim);
    throwErrorIf(mat->dim() != dim, "%s, Shape don't match.", func);
}

template<class real> inline
void prepVector(sq::VectorType<real> *vec, const sq::SizeType size, const char *func) {
    throwErrorIf(vec == NULL, "%s, vector is NULL.", func);
    if (vec->data == NULL)
        vec->resize(size);
    throwErrorIf(vec->size != size, "%s, Shape don't match.", func);
}

template<class real> inline
void validateScalar(real *sc, const char *func) {
    throwErrorIf(sc == NULL, "%s, Scalar is null.", func);
}


}

