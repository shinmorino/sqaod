#include "types.h"
#include "Traits.h"
#include <iostream>

using namespace quantd_cpu;

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
void SolverTraits<real>::denseGraphCalculate_E(real *E,
                                               const real *W, const char *x, int N) {
    const Eigen::Map<Matrix> eW((real*)W, N, N);
    const Matrix ex = utils<real>::bitsToMat(x, 1, N);
    Eigen::Map<Matrix> eE(E, 1, 1);
    eE = ex * (eW * ex.transpose());
}



template<class real>
void SolverTraits<real>::denseGraphBatchCalculate_E(real *E,
                                                    const real *W, const char *x,
                                                    int N, int nBatch) {
    const Eigen::Map<Matrix> eW((real*)W, N, N);
    const Matrix ex = utils<real>::bitsToMat(x, nBatch, N);
    Matrix eWx = eW * ex.transpose();
    Matrix prod = eWx.transpose().cwiseProduct(ex);
    Eigen::Map<Matrix>(E, nBatch, 1) = prod.rowwise().sum(); 
}



template<class real>
void SolverTraits<real>::denseGraphCalculate_hJc(real *h, real *J, real *c, const real *W, int N) {
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
void SolverTraits<real>::denseGraphCalculate_E_fromQbits(real *E,
                                                         const real *h, const real *J, real c,
                                                         const char *q, int N) {
    const Eigen::Map<Matrix> eh((real*)h, 1, N);
    const Eigen::Map<Matrix> eJ((real*)J, N, N);
    const Matrix eq = utils<real>::bitsToMat(q, N, 1);
    Eigen::Map<Matrix>(E, 1, 1) = Matrix::Constant(1, 1, c) + eh * eq + eq.transpose() * (eJ * eq);
}
    
    

template<class real>
void SolverTraits<real>::denseGraphBatchCalculate_E_fromQbits(real *E,
                                                              const real *h, const real *J, real c,
                                                              const char *q, int N, int nBatch) {
    const Eigen::Map<Matrix> eh((real*)h, 1, N);
    const Eigen::Map<Matrix> eJ((real*)J, N, N);
    Eigen::Map<Matrix> Ex(E, N, 1);

    const Matrix eq = utils<real>::bitsToMat(q, nBatch, N);
    Eigen::Map<Matrix> eE(E, 1, nBatch);

    Matrix tmp = eJ * eq.transpose();
    Matrix sum = tmp.cwiseProduct(eq.transpose()).colwise().sum(); /* batched dot product. */
    eE = Matrix::Constant(1, nBatch, c) + eh * eq.transpose() + sum;
}


/* rbm */

template<class real>
void SolverTraits<real>::rbmCalculate_E(real *E,
                                        const real *b0, const real *b1, const real *W,
                                        const char *x0, const char *x1,
                                        int N0, int N1) {
    const Eigen::Map<Matrix> eb0((real*)b0, N0, 1);
    const Eigen::Map<Matrix> eb1((real*)b1, N1, 1);
    const Eigen::Map<Matrix> eW((real*)W, N1, N0);
    const Matrix ex0 = utils<real>::bitsToMat(x0, N0, 1);
    const Matrix ex1 = utils<real>::bitsToMat(x1, N1, 1);
    Eigen::Map<Matrix> eE(E, 1, 1);
    eE = - eb0 * ex0 - eb1 * ex1 - ex1.transpose() * (eW * ex0);
}

template<class real>
void SolverTraits<real>::rbmBatchCalculate_E(real *E,
                                             const real *b0, const real *b1, const real *W,
                                             const char *x0, const char *x1,
                                             int N0, int N1, int nBatch0, int nBatch1) {
    const Eigen::Map<Matrix> eb0((real*)b0, N0, 1);
    const Eigen::Map<Matrix> eb1((real*)b1, N1, 1);
    const Eigen::Map<Matrix> eW((real*)W, N1, N0);
    const Matrix ex0 = utils<real>::bitsToMat(x0, nBatch0, N0);
    const Matrix ex1 = utils<real>::bitsToMat(x1, nBatch1, N1);
    Eigen::Map<Matrix> eE(E, nBatch1, nBatch0);
    eE = - ex1 * (eW * ex0.transpose());
    // Eigen::Matrix<real, 1, Eigen::Dynamic> v0 = eb0 * ex0.transpose();
    eE -= eb0 * ex0.transpose();
    eE -= (eb1 * ex1.transpose()).transpose();
    /* - np.matmul(b0, x0.T).reshape(1, iStep) - np.matmul(b1, x1.T).reshape(jStep, 1) \
        - np.matmul(x1, np.matmul(W, x0.T))
    */
}


#if 0


template<class real>
void SolverTraits<real>::denseGraphCalculate_hJc(real *h, real *J, real *c, const real *W, int N) {
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
void SolverTraits<real>::denseGraphCalculate_E_fromQbits(real *E,
                                                         const real *h, const real *J, real c,
                                                         const char *q, int N) {
    const Eigen::Map<Matrix> eh((real*)h, 1, N);
    const Eigen::Map<Matrix> eJ((real*)J, N, N);
    const Matrix eq = utils<real>::bitsToMat(q, N, 1);
    Eigen::Map<Matrix>(E, 1, 1) = Matrix::Constant(1, 1, c) + eh * eq + eq.transpose() * (eJ * eq);
}
    
    

template<class real>
void SolverTraits<real>::denseGraphBatchCalculate_E_fromQbits(real *E,
                                                              const real *h, const real *J, real c,
                                                              const char *q, int N, int nBatch) {
    const Eigen::Map<Matrix> eh((real*)h, 1, N);
    const Eigen::Map<Matrix> eJ((real*)J, N, N);
    Eigen::Map<Matrix> Ex(E, N, 1);

    const Matrix eq = utils<real>::bitsToMat(q, nBatch, N);
    Eigen::Map<Matrix> eE(E, 1, nBatch);

    Matrix tmp = eJ * eq.transpose();
    Matrix sum = tmp.cwiseProduct(eq.transpose()).colwise().sum(); /* batched dot product. */
    eE = Matrix::Constant(1, nBatch, c) + eh * eq.transpose() + sum;
}



def rbm_calculate_E(W, x0, x1) :
    return - np.dot(b0, x0) - np.dot(b1, x1) - np.dot(x1, np.matmul(W, x0))

def rbm_batch_calculate_E(W, x0, x1) :
    return - np.matmul(b0, x0.T).reshape(1, iStep) - np.matmul(b1, x1.T).reshape(jStep, 1) \
        - np.matmul(x1, np.matmul(W, x0.T))


def rbm_calculate_hJc(W) :
    N0 = W.shape[1]
    N1 = W.shape[0]
    
    c = 0.25 * np.sum(W) + 0.5 * (np.sum(b0) + np.sum(b1))
    J = 0.25 * W
    h0 = [(1. / 4.) * np.sum(W[:, i]) + 0.5 * b0[i] for i in range(0, N0)]
    h1 = [(1. / 4.) * np.sum(W[j]) + 0.5 * b1[j] for j in range(0, N1)]
    hlist = [h0, h1]

    return hlist, J, c

    
def rbm_calculate_E_from_qbits(h, J, c, q0, q1) :
    return - np.dot(h0, q0) - np.dot(h1, q1) - np.dot(q1[0], np.matmul(J, q0[0])) - c

def rbm_batch_calculate_E_from_qbits(h, J, c, q0, q1) :
    return - np.matmul(h0, q0.T).reshape(1, q0.shape[0]) \
        - np.matmul(h1, q1.T).reshape(1, q1.shape[1]) \
        - np.dot(q1[0], np.matmul(J, q0[0])) - c

#endif


template struct utils<double>;
template struct utils<float>;
template struct SolverTraits<double>;
template struct SolverTraits<float>;

