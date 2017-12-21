#include "Traits.h"
#include <iostream>
#include <float.h>



template<class real>
void sqaod::createBitsSequence(real *bits, int nBits, int bBegin, int bEnd) {
    for (int b = bBegin; b < bEnd; ++b) {
        for (int pos = nBits - 1; pos != -1; --pos)
            bits[pos] = ((b >> pos) & 1);
        bits += nBits;
    }
}

void sqaod::unpackBits(Bits *unpacked, const PackedBits packed, int N) {
    unpacked->resize(N);
    for (int pos = N - 1; pos != -1; --pos)
        (*unpacked)(pos) = (packed >> pos) & 1;
}

using namespace sqaod;

template<class real>
bool utils<real>::isSymmetric(const real *W, int N) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i)
            if (W[i + j * N]!= W[j + i * N])
                return false;
    }
    return true;
}


template<class real>
typename utils<real>::Matrix utils<real>::bitsToMat(const char *bits, int nRows, int nCols) {
    Matrix mat(nRows, nCols);
    for (int j = 0; j < nCols; ++j) {
        for (int i = 0; i < nRows; ++i)
            mat(i, j) = bits[i * nCols + j];
    }
    return mat;
}






template<class real>
void DGFuncs<real>::calculate_E(real *E,
                                const real *W, const char *x, int N) {
    const Eigen::Map<Matrix> eW((real*)W, N, N);
    const Matrix ex = utils<real>::bitsToMat(x, 1, N);
    Eigen::Map<Matrix> eE(E, 1, 1);
    eE = ex * (eW * ex.transpose());
}


template<class real>
void DGFuncs<real>::batchCalculate_E(real *E,
                                     const real *W, const char *x,
                                     int N, int nBatch) {
    const Eigen::Map<Matrix> eW((real*)W, N, N);
    const Matrix ex = utils<real>::bitsToMat(x, nBatch, N);
    Matrix eWx = eW * ex.transpose();
    Matrix prod = eWx.transpose().cwiseProduct(ex);
    Eigen::Map<Matrix>(E, nBatch, 1) = prod.rowwise().sum(); 
}



template<class real>
void DGFuncs<real>::calculate_hJc(real *h, real *J, real *c, const real *W, int N) {
    THROW_IF(!utils<real>::isSymmetric(W, N), "W is not symmetric.");
    
    Eigen::Map<Matrix> eW((real*)W, N, N);
    Eigen::Map<Matrix> eh(h, 1, N);
    Eigen::Map<Matrix> eJ(J, N, N);
    
    eh = real(0.5) * eW.colwise().sum();

    eJ = 0.25 * eW;
    real eJsum = eJ.sum();
    real diagSum = eJ.diagonal().sum();
    for (int i = 0; i < N; ++i)
        eJ(i, i) = real(0.);
    *c = eJsum + diagSum;
}

template<class real>
void DGFuncs<real>::calculate_E_fromQbits(real *E,
                                          const real *h, const real *J, real c,
                                          const char *q, int N) {
    const Eigen::Map<Matrix> eh((real*)h, 1, N);
    const Eigen::Map<Matrix> eJ((real*)J, N, N);
    const Matrix eq = utils<real>::bitsToMat(q, N, 1);
    *E = c + (eh * eq + eq.transpose() * (eJ * eq))(0, 0);
}

template<class real>
void DGFuncs<real>::calculate_E_fromQbits(real *E,
                                          const real *h, const real *J, real c,
                                          const real *q, int N) {
    const Eigen::Map<Matrix> eh((real*)h, 1, N);
    const Eigen::Map<Matrix> eJ((real*)J, N, N);
    const Eigen::Map<Matrix> eq((real*)q, N, 1);
    *E = c + (eh * eq + eq.transpose() * (eJ * eq))(0, 0);
}


template<class real>
void DGFuncs<real>::batchCalculate_E_fromQbits(real *E,
                                               const real *h, const real *J, real c,
                                               const char *q, int N, int nBatch) {
    const Matrix eq = utils<real>::bitsToMat(q, nBatch, N);
    batchCalculate_E_fromQbits(E, h, J, c, eq.data(), N, nBatch);
}


template<class real>
void DGFuncs<real>::batchCalculate_E_fromQbits(real *E,
                                               const real *h, const real *J, real c,
                                               const real *q, int N, int nBatch) {
    const Eigen::Map<Matrix> eh((real*)h, 1, N);
    const Eigen::Map<Matrix> eJ((real*)J, N, N);
    Eigen::Map<Matrix> Ex(E, N, 1);
    
    const Eigen::Map<Matrix> eq((real*)q, nBatch, N);
    Eigen::Map<Matrix> eE(E, 1, nBatch);
    
    Matrix tmp = eJ * eq.transpose();
    Matrix sum = tmp.cwiseProduct(eq.transpose()).colwise().sum(); /* batched dot product. */
    eE = eh * eq.transpose() + sum;
    eE.array() += c;
}


template<class real>
void DGFuncs<real>::batchSearch(real *E, PackedBitsArray *xList, const real *W, int N,
                                PackedBits xBegin, PackedBits xEnd) {
    const Eigen::Map<Matrix> eW((real*)W, N, N);
    batchSearch(E, xList, eW, xBegin, xEnd);
}


template<class real>
void DGFuncs<real>::batchSearch(real *E, PackedBitsArray *xList,
                                const Matrix &eW, PackedBits xBegin, PackedBits xEnd) {
    int nBatch = int(xEnd - xBegin);
    int N = eW.rows();

    real Emin = *E;
    Matrix eBitsSeq(nBatch, N);
    ColumnVector eEbatch(nBatch);

    createBitsSequence(eBitsSeq.data(), N, xBegin, xEnd);
    Matrix eWx = eW * eBitsSeq.transpose();
    Matrix prod = eWx.transpose().cwiseProduct(eBitsSeq);
    eEbatch = prod.rowwise().sum(); 
    /* FIXME: Parallelize */
    for (int idx = 0; idx < nBatch; ++idx) {
        if (eEbatch(idx) > Emin) {
            continue;
        }
        else if (eEbatch(idx) == Emin) {
            xList->push_back(xBegin + idx);
        }
        else {
            Emin = eEbatch(idx);
            xList->clear();
            xList->push_back(idx);
        }
    }
    *E = Emin;
}


/* rbm */

template<class real>
void BGFuncs<real>::calculate_E(real *E,
                                 const real *b0, const real *b1, const real *W,
                                 const char *x0, const char *x1,
                                 int N0, int N1) {
    const Eigen::Map<Matrix> eb0((real*)b0, 1, N0);
    const Eigen::Map<Matrix> eb1((real*)b1, 1, N1);
    const Eigen::Map<Matrix> eW((real*)W, N1, N0);
    const Matrix ex0 = utils<real>::bitsToMat(x0, N0, 1);
    const Matrix ex1 = utils<real>::bitsToMat(x1, N1, 1);
    Eigen::Map<Matrix> eE(E, 1, 1);
    Matrix prod = (eW * ex0);
    eE = eb0 * ex0 + eb1 * ex1 + ex1.transpose() * (eW * ex0);
}

template<class real>
void BGFuncs<real>::batchCalculate_E(real *E,
                                      const real *b0, const real *b1, const real *W,
                                      const char *x0, const char *x1,
                                      int N0, int N1, int nBatch0, int nBatch1) {
    const Eigen::Map<Matrix> eb0((real*)b0, 1, N0);
    const Eigen::Map<Matrix> eb1((real*)b1, 1, N1);
    const Eigen::Map<Matrix> eW((real*)W, N1, N0);
    const Matrix ex0 = utils<real>::bitsToMat(x0, nBatch0, N0);
    const Matrix ex1 = utils<real>::bitsToMat(x1, nBatch1, N1);
    Eigen::Map<Matrix> eE(E, nBatch1, nBatch0);

    RowVector ebx0 = eb0 * ex0.transpose();
    ColumnVector ebx1 = (eb1 * ex1.transpose()).transpose();
    eE.rowwise() = ebx0;
    eE.colwise() += ebx1;
    eE += ex1 * (eW * ex0.transpose());
}


template<class real>
void BGFuncs<real>::calculate_hJc(real *h0, real *h1, real *J, real *c,
                                   const real *b0, const real *b1, const real *W,
                                   int N0, int N1) {
    const Eigen::Map<Matrix> eb0((real*)b0, N0, 1), eb1((real*)b1, N1, 1), eW((real*)W, N1, N0);
    Eigen::Map<Matrix> eh0(h0, N0, 1), eh1(h1, N1, 1);
    Eigen::Map<Matrix> eJ(J, N1, N0);

    eJ = real(0.25) * eW;
    eh0 = real(0.25) * eW.colwise().sum().transpose() + real(0.5) * eb0;
    eh1 = real(0.25) * eW.rowwise().sum() + real(0.5) * eb1;
    *c = real(0.25) * eW.sum() + real(0.5) * (eb0.sum() + eb1.sum());
}


template<class real>
void BGFuncs<real>::calculate_hJc(RowVector *h0, RowVector *h1, Matrix *J, real *c,
                                  const Matrix &b0, const Matrix &b1, const Matrix &W) {

    *J = real(0.25) * W;
    *h0 = real(0.25) * W.colwise().sum().transpose() + real(0.5) * b0;
    *h1 = real(0.25) * W.rowwise().sum() + real(0.5) * b1;
    *c = real(0.25) * W.sum() + real(0.5) * (b0.sum() + b1.sum());
}


template<class real>
void BGFuncs<real>::
calculate_E_fromQbits(real *E,
                      const real *h0, const real *h1, const real *J, real c,
                      const char *q0, const char *q1,
                      int N0, int N1) {
    const Eigen::Map<Matrix> eh0((real*)h0, 1, N0), eh1((real*)h1, 1, N1);
    const Eigen::Map<Matrix> eJ((real*)J, N1, N0);
    const Matrix eq0 = utils<real>::bitsToMat(q0, N0, 1);
    const Matrix eq1 = utils<real>::bitsToMat(q1, N1, 1);
    *E = (eh0 * eq0 + eh1 * eq1 + eq1.transpose() * (eJ * eq0))(0, 0) + c;
}
    

template<class real>
void BGFuncs<real>::
batchCalculate_E_fromQbits(real *E,
                           const real *h0, const real *h1, const real *J, real c,
                           const char *q0, const char *q1,
                           int N0, int N1, int nBatch0, int nBatch1) {
    const Eigen::Map<Matrix> eh0((real*)h0, 1, N0);
    const Eigen::Map<Matrix> eh1((real*)h1, 1, N1);
    const Eigen::Map<Matrix> eJ((real*)J, N1, N0);
    const Matrix eq0 = utils<real>::bitsToMat(q0, nBatch0, N0);
    const Matrix eq1 = utils<real>::bitsToMat(q1, nBatch1, N1);
    Eigen::Map<Matrix> eE(E, nBatch1, nBatch0);

    RowVector ehq0 = eh0 * eq0.transpose();
    ColumnVector ehq1 = (eh1 * eq1.transpose()).transpose();
    eE.rowwise() = ehq0;
    eE.colwise() += ehq1;
    eE += eq1 * (eJ * eq0.transpose());
    eE.array() += c;
}

#if 0

def rbm_batch_calculate_E_from_qbits(h, J, c, q0, q1) :
    return - np.matmul(h0, q0.T).reshape(1, q0.shape[0]) \
        - np.matmul(h1, q1.T).reshape(1, q1.shape[1]) \
        - np.dot(q1[0], np.matmul(J, q0[0])) - c

#endif

template<class real>
void BGFuncs<real>::batchSearch(real *E, PackedBitsPairArray *xPairs,
                                const RowVector &b0, const RowVector &b1, const Matrix &W,
                                PackedBits xBegin0, PackedBits xEnd0,
                                PackedBits xBegin1, PackedBits xEnd1) {
    int nBatch0 = int(xEnd0 - xBegin0);
    int nBatch1 = int(xEnd1 - xBegin1);

    real Emin = *E;
    int N0 = W.cols();
    int N1 = W.rows();
    Matrix eBitsSeq0(nBatch0, N0);
    Matrix eBitsSeq1(nBatch1, N1);

    createBitsSequence(eBitsSeq0.data(), N0, xBegin0, xEnd0);
    createBitsSequence(eBitsSeq1.data(), N1, xBegin1, xEnd1);
    
    Matrix eEBatch = eBitsSeq1 * (W * eBitsSeq0.transpose());
    eEBatch.rowwise() += b0 * eBitsSeq0.transpose();
    eEBatch.colwise() += (b1 * eBitsSeq1.transpose()).transpose();
    
    /* FIXME: Parallelize */
    for (int idx1 = 0; idx1 < nBatch1; ++idx1) {
        for (int idx0 = 0; idx0 < nBatch0; ++idx0) {
            real Etmp = eEBatch(idx1, idx0);
            if (Etmp > Emin) {
                continue;
            }
            else if (Etmp == Emin) {
                xPairs->push_back(PackedBitsPairArray::value_type(xBegin0 + idx0, xBegin1 + idx1));
            }
            else {
                Emin = Etmp;
                xPairs->clear();
                xPairs->push_back(PackedBitsPairArray::value_type(xBegin0 + idx0, xBegin1 + idx1));
            }
        }
    }
    *E = Emin;
}
    

template struct utils<double>;
template struct utils<float>;
template struct DGFuncs<double>;
template struct DGFuncs<float>;
template struct BGFuncs<double>;
template struct BGFuncs<float>;

template
void ::sqaod::createBitsSequence(double *bits, int nBits, int bBegin, int bEnd);
template
void ::sqaod::createBitsSequence(float *bits, int nBits, int bBegin, int bEnd);
template
void ::sqaod::createBitsSequence(char *bits, int nBits, int bBegin, int bEnd);
