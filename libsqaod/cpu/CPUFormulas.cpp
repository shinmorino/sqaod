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
    
    const EigenMappedMatrix eW(W.map());
    EigenMappedColumnVector ex(x.mapToColumnVector()); 
    EigenMappedMatrix eE(E, 1, 1);
    eE = ex.transpose() * (eW * ex);
}


template<class real>
void DGFuncs<real>::calculate_E(Vector *E, const Matrix &W, const Matrix &x) {
    prepVector(E, x.rows, __func__);
    quboShapeCheck(W, x, __func__);

    EigenMappedMatrix ex(x.map());
    EigenMatrix eWx = W.map() * ex.transpose();
    EigenMatrix prod = eWx.transpose().cwiseProduct(ex);
    EigenMappedColumnVector eE(E->mapToColumnVector());
    eE = prod.rowwise().sum(); 
}


template<class real>
void DGFuncs<real>::calculate_hJc(Vector *h, Matrix *J, real *c, const Matrix &W) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric, %s.");
    prepVector(h, W.rows, __func__);
    prepMatrix(J, W.dim(), __func__);
    validateScalar(c, __func__);
    
    const EigenMappedMatrix eW = W.map();
    EigenMappedMatrix eJ(J->map());
    EigenMappedRowVector eh(h->mapToRowVector());
    
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

    const EigenMappedRowVector eh(h.mapToRowVector());
    const EigenMappedMatrix eJ(J.map());
    const EigenMappedColumnVector eq(q.mapToColumnVector());
    *E = c + (eh * eq + eq.transpose() * (eJ * eq))(0, 0);
}

template<class real>
void DGFuncs<real>::calculate_E(Vector *E,
                                const Vector &h, const Matrix &J, real c, const Matrix &q) {
    validateScalar(E, __func__);
    isingModelShapeCheck(h, J, c, q, __func__);
    
    const EigenMappedRowVector eh(h.mapToRowVector());
    const EigenMappedMatrix eJ(J.map()), eq(q.map());
    EigenMappedColumnVector eE(E->mapToColumnVector());
    
    EigenMatrix tmp = eJ * eq.transpose();
    /* FIXME: further optimization might be required. */
    EigenMatrix sum = tmp.cwiseProduct(eq.transpose()).colwise().sum(); /* batched dot product. */
    eE = eh * eq.transpose() + sum;
    eE.array() += c;
}


template<class real>
void DGFuncs<real>::batchSearch(real *E, PackedBitsArray *xList,
                                const Matrix &W, PackedBits xBegin, PackedBits xEnd) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric, %s.");

    const EigenMappedMatrix eW(W.map());
    int nBatch = int(xEnd - xBegin);
    int N = eW.rows();

    real Emin = *E;
    EigenMatrix eBitsSeq(nBatch, N);
    EigenMatrix eEbatch(nBatch, 1);

    createBitsSequence(eBitsSeq.data(), N, xBegin, xEnd);
    EigenMatrix eWx = eW * eBitsSeq.transpose();
    EigenMatrix prod = eWx.transpose().cwiseProduct(eBitsSeq);
    eEbatch = prod.rowwise().sum(); 
    /* FIXME: Parallelize */
    for (int idx = 0; idx < nBatch; ++idx) {
        if (eEbatch(idx) > Emin) {
            continue;
        }
        else if (eEbatch(idx) == Emin) {
            xList->pushBack(xBegin + idx);
        }
        else {
            Emin = eEbatch(idx);
            xList->clear();
            xList->pushBack(idx);
        }
    }
    *E = Emin;
}


/* rbm */

template<class real>
void BGFuncs<real>::calculate_E(real *E,
                                const Vector &b0, const Vector &b1, const Matrix &W,
                                const Vector &x0, const Vector &x1) {
    quboShapeCheck(b0, b1, W, x0, x1, __func__);
    validateScalar(E, __func__);
    
    const EigenMappedRowVector eb0(b0.mapToRowVector()), eb1(b1.mapToRowVector());
    const EigenMappedMatrix eW(W.map());
    const EigenMappedColumnVector ex0(x0.mapToColumnVector()), ex1(x1.mapToColumnVector());
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
    EigenMappedRowVector eE(E->mapToRowVector());
    const EigenMappedRowVector eb0(b0.mapToRowVector()), eb1(b1.mapToRowVector());
    const EigenMappedMatrix eW(W.map()), ex0(x0.map()), ex1(x1.map());

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
    
    EigenMappedMatrix eE(E->map());
    const EigenMappedRowVector eb0(b0.mapToRowVector()), eb1(b1.mapToRowVector());
    const EigenMappedMatrix eW(W.map()), ex0(x0.map()), ex1(x1.map());

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

    const EigenMappedRowVector eb0(b0.mapToRowVector()), eb1(b1.mapToRowVector());
    const EigenMappedMatrix eW(W.map());
    EigenMappedRowVector eh0(h0->mapToRowVector()), eh1(h1->mapToRowVector());
    EigenMappedMatrix eJ(J->map());
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
    
    const EigenMappedRowVector eh0(h0.mapToRowVector()), eh1(h1.mapToRowVector());
    const EigenMappedMatrix eJ(J.map());
    const EigenMappedColumnVector eq0(q0.mapToColumnVector()), eq1(q1.mapToColumnVector());
    *E = (eh0 * eq0 + eh1 * eq1 + eq1.transpose() * (eJ * eq0))(0, 0) + c;
}


template<class real>
void BGFuncs<real>::calculate_E(Vector *E,
                                const Vector &h0, const Vector &h1, const Matrix &J, real c,
                                const Matrix &q0, const Matrix &q1) {
    isingModelShapeCheck(h0, h1, J, c, q0, q1, __func__);
    prepVector(E, q0.rows, __func__);

    const EigenMappedRowVector eh0(h0.mapToRowVector()), eh1(h1.mapToRowVector());
    const EigenMappedMatrix eJ(J.map());
    const EigenMappedMatrix eq0(q0.map()), eq1(q1.map());
    EigenMappedRowVector eE(E->mapToRowVector());

    EigenMatrix tmp = eJ * eq0.transpose();
    /* FIXME: further optimization might be required. */
    eE = tmp.cwiseProduct(eq1.transpose()).colwise().sum(); /* batched dot product. */
    eE += eh0 * eq0.transpose();
    eE += eh1 * eq1.transpose();
    eE.array() += c;
}


template<class real>
void BGFuncs<real>::batchSearch(real *E, PackedBitsPairArray *xPairs,
    const Vector &b0, const Vector &b1, const Matrix &W,
    PackedBits xBegin0, PackedBits xEnd0,
    PackedBits xBegin1, PackedBits xEnd1) {

    batchSearch(E, xPairs, b0.mapToRowVector(), b1.mapToRowVector(), W.map(), xBegin0, xEnd0, xBegin1, xEnd1);
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

template<class real>
void BGFuncs<real>::batchSearch(real *E, PackedBitsPairArray *xPairs,
                                const EigenRowVector &b0, const EigenRowVector &b1, const EigenMatrix &W,
                                PackedBits xBegin0, PackedBits xEnd0,
                                PackedBits xBegin1, PackedBits xEnd1) {
    int nBatch0 = int(xEnd0 - xBegin0);
    int nBatch1 = int(xEnd1 - xBegin1);

    real Emin = *E;
    int N0 = W.cols();
    int N1 = W.rows();
    EigenMatrix eBitsSeq0(nBatch0, N0);
    EigenMatrix eBitsSeq1(nBatch1, N1);

    createBitsSequence(eBitsSeq0.data(), N0, xBegin0, xEnd0);
    createBitsSequence(eBitsSeq1.data(), N1, xBegin1, xEnd1);
    
    EigenMatrix eEBatch = eBitsSeq1 * (W * eBitsSeq0.transpose());
    eEBatch.rowwise() += (b0 * eBitsSeq0.transpose()).row(0);
    eEBatch.colwise() += (b1 * eBitsSeq1.transpose()).transpose().col(0);
    
    /* FIXME: Parallelize */
    for (int idx1 = 0; idx1 < nBatch1; ++idx1) {
        for (int idx0 = 0; idx0 < nBatch0; ++idx0) {
            real Etmp = eEBatch(idx1, idx0);
            if (Etmp > Emin) {
                continue;
            }
            else if (Etmp == Emin) {
                xPairs->pushBack(PackedBitsPairArray::ValueType(xBegin0 + idx0, xBegin1 + idx1));
            }
            else {
                Emin = Etmp;
                xPairs->clear();
                xPairs->pushBack(PackedBitsPairArray::ValueType(xBegin0 + idx0, xBegin1 + idx1));
            }
        }
    }
    *E = Emin;
}
    

template struct DGFuncs<double>;
template struct DGFuncs<float>;
template struct BGFuncs<double>;
template struct BGFuncs<float>;
