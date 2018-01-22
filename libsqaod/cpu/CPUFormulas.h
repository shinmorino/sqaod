#ifndef CPUFORMULAS_H__
#define CPUFORMULAS_H__

#include <common/Common.h>

namespace sqaod {
    
template<class real>
struct DGFuncs {
    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenMappedMatrixType<real> EigenMappedMatrix;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;
    typedef EigenMappedRowVectorType<real> EigenMappedRowVector;
    typedef EigenMappedColumnVectorType<real> EigenMappedColumnVector;
    
    static
    void calculate_E(real *E, const Matrix &W, const Vector &x);
    
    static
    void calculate_E(Vector *E, const Matrix &W, const Matrix &x);
    
    static
    void calculate_hJc(Vector *h, Matrix *J, real *c, const Matrix &W);
    
    static
    void calculate_E(real *E,
                     const Vector &h, const Matrix &J, real c, const Vector &q);

    static
    void calculate_E(Vector *E,
                     const Vector &h, const Matrix &J, real c, const Matrix &q);
    
    static
    void batchSearch(real *E, PackedBitsArray *xList,
                     const Matrix &W, PackedBits xBegin, PackedBits xEnd);
};
    
template<class real>
struct BGFuncs {
    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenMappedMatrixType<real> EigenMappedMatrix;
    typedef EigenRowVectorType<real> EigenRowVector;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;
    typedef EigenMappedRowVectorType<real> EigenMappedRowVector;
    typedef EigenMappedColumnVectorType<real> EigenMappedColumnVector;

    static
    void calculate_E(real *E,
                     const Vector &b0, const Vector &b1, const Matrix &W,
                     const Vector &x0, const Vector &x1);
    
    static
    void calculate_E(Vector *E,
                     const Vector &b0, const Vector &b1, const Matrix &W,
                     const Matrix &x0, const Matrix &x1);

    static
    void calculate_E_2d(Matrix *E,
                        const Vector &b0, const Vector &b1, const Matrix &W,
                        const Matrix &x0, const Matrix &x1);
    
    static
    void calculate_hJc(Vector *h0, Vector *h1, Matrix *J, real *c,
                       const Vector &b0, const Vector &b1, const Matrix &W);

    static
    void calculate_E(real *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Vector &q0, const Vector &q1);

    static
    void calculate_E(Vector *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Matrix &q0, const Matrix &q1);

    static
    void batchSearch(real *E, PackedBitsPairArray *xList,
                     const Vector &b0, const Vector &b1, const Matrix &W,
                     PackedBits xBegin0, PackedBits xEnd0,
                     PackedBits xBegin1, PackedBits xEnd1);
    
    /* Eigen ver */
    static
    void calculate_hJc(EigenRowVector *h0, EigenRowVector *h1, EigenMatrix *J, real *c,
                       const EigenRowVector &b0, const EigenRowVector &b1, const EigenMatrix &W);

    static
    void batchSearch(real *E, PackedBitsPairArray *xList,
                     const EigenRowVector &b0, const EigenRowVector &b1, const EigenMatrix &W,
                     PackedBits xBegin0, PackedBits xEnd0,
                     PackedBits xBegin1, PackedBits xEnd1);
};


}

#endif
