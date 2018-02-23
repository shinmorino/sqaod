#include "CPUFormulas.h"
#include "CPUObjectPrep.h"
#include <iostream>
#include <float.h>


using namespace sqaod;
using namespace sqaod_cpu;


template<class real>
void DGFuncs<real>::calculate_E(real *E,
                                const Matrix &W, const Vector &x) {
    validateScalar(E, __func__);
    quboShapeCheck(W, x, __func__);
    
    const EigenMappedMatrix eW(mapTo(W));
    EigenMappedColumnVector ex(mapToColumnVector(x)); 
    EigenMappedMatrix eE(E, 1, 1);
    eE = ex.transpose() * (eW * ex);
}


template<class real>
void DGFuncs<real>::calculate_E(Vector *E, const Matrix &W, const Matrix &x) {
    prepVector(E, x.rows, __func__);
    quboShapeCheck(W, x, __func__);

    EigenMappedMatrix ex(mapTo(x));
    EigenMatrix eWx = mapTo(W) * ex.transpose();
    EigenMatrix prod = eWx.transpose().cwiseProduct(ex);
    EigenMappedColumnVector eE(mapToColumnVector(*E));
    eE = prod.rowwise().sum(); 
}


template<class real>
void DGFuncs<real>::calculate_hJc(Vector *h, Matrix *J, real *c, const Matrix &W) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric, %s.");
    prepVector(h, W.rows, __func__);
    prepMatrix(J, W.dim(), __func__);
    validateScalar(c, __func__);

    const EigenMappedMatrix eW(mapTo(W));
    EigenMappedMatrix eJ(mapTo(*J));
    EigenMappedRowVector eh(mapToRowVector(*h));
    
    eh = real(0.5) * eW.colwise().sum();

    eJ = 0.25 * eW;
    real eJsum = eJ.sum();
    real diagSum = eJ.diagonal().sum();
    int N = W.rows;
    for (int i = 0; i < N; ++i)
        eJ(i, i) = real(0.);
    *c = eJsum + diagSum;
}

template<class real>
void DGFuncs<real>::calculate_E(real *E,
                                const Vector &h, const Matrix &J, real c, const Vector &q) {
    validateScalar(E, __func__);
    isingModelShapeCheck(h, J, c, q, __func__);

    const EigenMappedRowVector eh(mapToRowVector(h));
    const EigenMappedMatrix eJ(mapTo(J));
    const EigenMappedColumnVector eq(mapToColumnVector(q));
    *E = c + (eh * eq + eq.transpose() * (eJ * eq))(0, 0);
}

template<class real>
void DGFuncs<real>::calculate_E(Vector *E,
                                const Vector &h, const Matrix &J, real c, const Matrix &q) {
    isingModelShapeCheck(h, J, c, q, __func__);
    prepVector(E, q.rows, __func__);
    
    const EigenMappedRowVector eh(mapToRowVector(h));
    const EigenMappedMatrix eJ(mapTo(J)), eq(mapTo(q));
    EigenMappedColumnVector eE(mapToColumnVector(*E));
    
    EigenMatrix tmp = eJ * eq.transpose();
    /* FIXME: further optimization might be required. */
    EigenMatrix sum = tmp.cwiseProduct(eq.transpose()).colwise().sum(); /* batched dot product. */
    eE = eh * eq.transpose() + sum;
    eE.array() += c;
}


/* rbm */

template<class real>
void BGFuncs<real>::calculate_E(real *E,
                                const Vector &b0, const Vector &b1, const Matrix &W,
                                const Vector &x0, const Vector &x1) {
    quboShapeCheck(b0, b1, W, x0, x1, __func__);
    validateScalar(E, __func__);
    
    const EigenMappedRowVector eb0(mapToRowVector(b0)), eb1(mapToRowVector(b1));
    const EigenMappedMatrix eW(mapTo(W));
    const EigenMappedColumnVector ex0(mapToColumnVector(x0)), ex1(mapToColumnVector(x1));
    EigenMatrix prod = (eW * ex0);
    *E = (eb0 * ex0 + eb1 * ex1 + ex1.transpose() * (eW * ex0))(0, 0);
}

template<class real>
void BGFuncs<real>::calculate_E(Vector *E,
                                const Vector &b0, const Vector &b1, const Matrix &W,
                                const Matrix &x0, const Matrix &x1) {
    quboShapeCheck(b0, b1, W, x0, x1, __func__);
    prepVector(E, x1.rows, __func__);

    throwErrorIf(E == NULL, "E is NULL.");
    EigenMappedRowVector eE(mapToRowVector(*E));
    const EigenMappedRowVector eb0(mapToRowVector(b0)), eb1(mapToRowVector(b1));
    const EigenMappedMatrix eW(mapTo(W)), ex0(mapTo(x0)), ex1(mapTo(x1));

    EigenMatrix tmp = eW * ex0.transpose();
    /* FIXME: further optimization might be required. */
    eE = tmp.cwiseProduct(ex1.transpose()).colwise().sum(); /* batched dot product. */
    eE += eb0 * ex0.transpose();
    eE += eb1 * ex1.transpose();
}

template<class real>
void BGFuncs<real>::calculate_E_2d(Matrix *E,
                                   const Vector &b0, const Vector &b1, const Matrix &W,
                                   const Matrix &x0, const Matrix &x1) {    
    quboShapeCheck_2d(b0, b1, W, x0, x1, __func__);
    prepMatrix(E, Dim(x1.rows, x0.rows), __func__);
    
    EigenMappedMatrix eE(mapTo(*E));
    const EigenMappedRowVector eb0(mapToRowVector(b0)), eb1(mapToRowVector(b1));
    const EigenMappedMatrix eW(mapTo(W)), ex0(mapTo(x0)), ex1(mapTo(x1));

    EigenMatrix ebx0 = eb0 * ex0.transpose();
    EigenMatrix ebx1 = (eb1 * ex1.transpose()).transpose(); /* FIXME: reduce transpose */
    eE.rowwise() = ebx0.row(0);
    eE.colwise() += ebx1.col(0);
    eE += ex1 * (eW * ex0.transpose());
}


template<class real>
void BGFuncs<real>::calculate_hJc(Vector *h0, Vector *h1, Matrix *J, real *c,
                                  const Vector &b0, const Vector &b1, const Matrix &W) {
    
    quboShapeCheck(b0, b1, W, __func__);

    prepVector(h0, b0.size, __func__);
    prepVector(h1, b1.size, __func__);
    prepMatrix(J, W.dim(), __func__);
    validateScalar(c, __func__);
    quboShapeCheck(*h0, *h1, *J, __func__);

    const EigenMappedRowVector eb0(mapToRowVector(b0)), eb1(mapToRowVector(b1));
    const EigenMappedMatrix eW(mapTo(W));
    EigenMappedRowVector eh0(mapToRowVector(*h0)), eh1(mapToRowVector(*h1));
    EigenMappedMatrix eJ(mapTo(*J));
    // calculate_hJc(&eh0, &eh1, &eJ, c, eb0, eb1, W);
    
    eJ = real(0.25) * eW;
    eh0 = real(0.25) * eW.colwise().sum()+ real(0.5) * eb0;
    eh1 = real(0.25) * eW.rowwise().sum().transpose() + real(0.5) * eb1;
    *c = real(0.25) * eW.sum() + real(0.5) * (eb0.sum() + eb1.sum());
}


template<class real>
void BGFuncs<real>::calculate_E(real *E,
                                const Vector &h0, const Vector &h1, const Matrix &J, real c,
                                const Vector &q0, const Vector &q1) {
    isingModelShapeCheck(h0, h1, J, c, q0, q1, __func__);
    validateScalar(E, __func__);
    
    const EigenMappedRowVector eh0(mapToRowVector(h0)), eh1(mapToRowVector(h1));
    const EigenMappedMatrix eJ(mapTo(J));
    const EigenMappedColumnVector eq0(mapToColumnVector(q0)), eq1(mapToColumnVector(q1));
    *E = (eh0 * eq0 + eh1 * eq1 + eq1.transpose() * (eJ * eq0))(0, 0) + c;
}


template<class real>
void BGFuncs<real>::calculate_E(Vector *E,
                                const Vector &h0, const Vector &h1, const Matrix &J, real c,
                                const Matrix &q0, const Matrix &q1) {
    isingModelShapeCheck(h0, h1, J, c, q0, q1, __func__);
    prepVector(E, q0.rows, __func__);

    const EigenMappedRowVector eh0(mapToRowVector(h0)), eh1(mapToRowVector(h1));
    const EigenMappedMatrix eJ(mapTo(J));
    const EigenMappedMatrix eq0(mapTo(q0)), eq1(mapTo(q1));
    EigenMappedRowVector eE(mapToRowVector(*E));

    EigenMatrix tmp = eJ * eq0.transpose();
    /* FIXME: further optimization might be required. */
    eE = tmp.cwiseProduct(eq1.transpose()).colwise().sum(); /* batched dot product. */
    eE += eh0 * eq0.transpose();
    eE += eh1 * eq1.transpose();
    eE.array() += c;
}


/* Eigen versions */

template<class real>
void BGFuncs<real>::calculate_hJc(EigenRowVector *h0, EigenRowVector *h1, EigenMatrix *J, real *c,
                                  const EigenRowVector &b0, const EigenRowVector &b1, const EigenMatrix &W) {

    *J = real(0.25) * W;
    *h0 = real(0.25) * W.colwise().sum().transpose() + real(0.5) * b0;
    *h1 = real(0.25) * W.rowwise().sum() + real(0.5) * b1;
    *c = real(0.25) * W.sum() + real(0.5) * (b0.sum() + b1.sum());
}


template struct DGFuncs<double>;
template struct DGFuncs<float>;
template struct BGFuncs<double>;
template struct BGFuncs<float>;
