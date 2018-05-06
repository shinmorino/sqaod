#pragma once

#include <sqaodc/sqaodc.h>
#include <sqaodc/cpu/SharedFormulas.h>

namespace sqaod_cpu {

template<class real>
struct CPUDenseGraphFormulas : sqaod::DenseGraphFormulas<real> {
    
    typedef sqaod_cpu::DGFuncs<real> Formulas;
    typedef typename Formulas::Matrix Matrix;
    typedef typename Formulas::Vector Vector;
    
    virtual ~CPUDenseGraphFormulas() { }
    
    virtual
    void calculate_E(real *E, const Matrix &W, const Vector &x);
    
    virtual
    void calculate_E(Vector *E, const Matrix &W, const Matrix &x);
    
    virtual
    void calculateHamiltonian(Vector *h, Matrix *J, real *c, const Matrix &W);
    
    virtual
    void calculate_E(real *E,
                     const Vector &h, const Matrix &J, real c, const Vector &q);

    virtual
    void calculate_E(Vector *E,
                     const Vector &h, const Matrix &J, real c, const Matrix &q);
    
};


template<class real>
struct CPUBipartiteGraphFormulas : sqaod::BipartiteGraphFormulas<real> {

    typedef sqaod_cpu::BGFuncs<real> Formulas;
    typedef typename Formulas::Matrix Matrix;
    typedef typename Formulas::Vector Vector;
    
    virtual ~CPUBipartiteGraphFormulas() { }

    virtual
    void calculate_E(real *E,
                     const Vector &b0, const Vector &b1, const Matrix &W,
                     const Vector &x0, const Vector &x1);
    
    virtual
    void calculate_E(Vector *E,
                     const Vector &b0, const Vector &b1, const Matrix &W,
                     const Matrix &x0, const Matrix &x1);

    virtual
    void calculate_E_2d(Matrix *E,
                        const Vector &b0, const Vector &b1, const Matrix &W,
                        const Matrix &x0, const Matrix &x1);
    
    virtual
    void calculateHamiltonian(Vector *h0, Vector *h1, Matrix *J, real *c,
                              const Vector &b0, const Vector &b1, const Matrix &W);

    virtual
    void calculate_E(real *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Vector &q0, const Vector &q1);

    virtual
    void calculate_E(Vector *E,
                     const Vector &h0, const Vector &h1, const Matrix &J, real c,
                     const Matrix &q0, const Matrix &q1);
    
};


}
