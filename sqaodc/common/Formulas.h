#pragma once

#include <sqaodc/common/Common.h>

namespace sqaod {

template<class real>
struct DenseGraphFormulas : NullBase {
    
    typedef sqaod::MatrixType<real> Matrix;
    typedef sqaod::VectorType<real> Vector;

    virtual ~DenseGraphFormulas() WAR_VC_NOTHROW { }
    
    virtual
    void calculate_E(real *E, const Matrix &W, const Vector &x) = 0;
    
    virtual
    void calculate_E(Vector *E, const Matrix &W, const Matrix &x) = 0;
    
    virtual
    void calculateHamiltonian(Vector *h, Matrix *J, real *c, const Matrix &W) = 0;
    
    virtual
    void calculate_E(real *E,
                     const Vector &h, const Matrix &J, real c, const Vector &q) = 0;

    virtual
    void calculate_E(Vector *E,
                     const Vector &h, const Matrix &J, real c, const Matrix &q) = 0;
    
};
    
template<class real>
struct BipartiteGraphFormulas : NullBase {
    typedef sqaod::MatrixType<real> Matrix;
    typedef sqaod::VectorType<real> Vector;

    virtual ~BipartiteGraphFormulas() WAR_VC_NOTHROW { }

    virtual
    void calculate_E(real *E,
                     const Vector &b0, const Vector &b1, const Matrix &W,
                     const Vector &x0, const Vector &x1) = 0;
    
    virtual
    void calculate_E(Vector *E,
                     const Vector &b0, const Vector &b1, const Matrix &W,
                     const Matrix &x0, const Matrix &x1) = 0;

    virtual
    void calculate_E_2d(Matrix *E,
                        const Vector &b0, const Vector &b1, const Matrix &W,
                        const Matrix &x0, const Matrix &x1) = 0;
    
    virtual
    void calculateHamiltonian(Vector *h0, Vector *h1, Matrix *J, real *c,
                              const Vector &b0, const Vector &b1, const Matrix &W) = 0;

    virtual
    void calculate_E(real *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Vector &q0, const Vector &q1) = 0;

    virtual
    void calculate_E(Vector *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Matrix &q0, const Matrix &q1) = 0;
    
};

}
