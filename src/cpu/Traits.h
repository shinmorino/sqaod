#ifndef TRAITS_H__
#define TRAITS_H__

#include <Eigen/Core>


#define THROW_IF(cond, msg) if (cond) throw std::runtime_error(msg);

namespace quantd_cpu {

enum SolverDir {
    solverMinimize,
    solverMaximize
};

    
template<class real>
void createBitsSequence(real *bits, int nBits, int bBegin, int bEnd);

    
template<class real>
struct utils {
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

    static
    bool isSymmetric(const real *W, int N);

    static
    Matrix bitsToMat(const char *bits, int nRows, int nCols);

};
    
template<class real>
struct DGFuncs {
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<real, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> ColumnVector;
    
    static
    void calculate_E(real *E, const real *W, const char *x, int N);
    
    static
    void batchCalculate_E(real *E, const real *W, const char *x, int N, int nBatch);
    
    static
    void calculate_hJc(real *h, real *J, real *c, const real *W, int N);
    
    static
    void calculate_E_fromQbits(real *E,
                               const real *h, const real *J, real c, const char *q,
                               int N);

    static
    void calculate_E_fromQbits(real *E,
                               const real *h, const real *J, real c, const real *q,
                               int N);
    
    static
    void batchCalculate_E_fromQbits(real *E,
                                    const real *h, const real *J, real c, const char *q,
                                    int N, int nBatch);

    static
    void batchSearch(real *E, char *x, const real *W, int N, int xBegin, int xEnd);

};
    
template<class real>
struct RBMFuncs {
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<real, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> ColumnVector;

    static
    void calculate_E(real *E,
                     const real *b0, const real *b1, const real *W,
                     const char *x0, const char *x1,
                     int N0, int N1);
    
    static
    void batchCalculate_E(real *E,
                          const real *b0, const real *b1, const real *W,
                          const char *x0, const char *x1,
                          int N0, int N1, int nBatch0, int nBatch1);
    
    static
    void calculate_hJc(real *h0, real *h1, real *J, real *c,
                       const real *b0, const real *b1, const real *W,
                       int N0, int N1);
    
    static
    void calculate_E_fromQbits(real *E,
                               const real *h0, const real *h1, const real *J, real c,
                               const char *q0, const char *q1,
                               int N0, int N1);
    
    static
    void batchCalculate_E_fromQbits(real *E,
                                    const real *h0, const real *h1, const real *J, real c,
                                    const char *q0, const char *q1,
                                    int N0, int N1, int nBatch0, int nBatch1);
    
    
};


}

#endif
