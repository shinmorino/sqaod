#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/common/EigenBridge.h>

namespace sqaod_cpu {

namespace sq = sqaod;

template<class real>
struct DGFuncs {
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;
    typedef sq::EigenMatrixType<real> EigenMatrix;
    typedef sq::EigenMappedMatrixType<real> EigenMappedMatrix;
    typedef sq::EigenMappedRowVectorType<real> EigenMappedRowVector;
    typedef sq::EigenMappedColumnVectorType<real> EigenMappedColumnVector;
    
    static
    void calculate_E(real *E, const Matrix &W, const Vector &x);
    
    static
    void calculate_E(Vector *E, const Matrix &W, const Matrix &x);
    
    static
    void calculateHamiltonian(Vector *h, Matrix *J, real *c, const Matrix &W);
    
    static
    void calculate_E(real *E,
                     const Vector &h, const Matrix &J, real c, const Vector &q);

    static
    void calculate_E(Vector *E,
                     const Vector &h, const Matrix &J, real c, const Matrix &q);
    
};
    
template<class real>
struct BGFuncs {
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;
    typedef sq::EigenMatrixType<real> EigenMatrix;
    typedef sq::EigenRowVectorType<real> EigenRowVector;
    typedef sq::EigenMappedMatrixType<real> EigenMappedMatrix;
    typedef sq::EigenMappedRowVectorType<real> EigenMappedRowVector;
    typedef sq::EigenMappedColumnVectorType<real> EigenMappedColumnVector;

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
    void calculateHamiltonian(Vector *h0, Vector *h1, Matrix *J, real *c,
                              const Vector &b0, const Vector &b1, const Matrix &W);

    static
    void calculate_E(real *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Vector &q0, const Vector &q1);

    static
    void calculate_E(Vector *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Matrix &q0, const Matrix &q1);
    
};

}
