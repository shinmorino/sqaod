#include "CPUFormulas.h"
#include <sqaodc/common/internal/ShapeChecker.h>

namespace sqcpu = sqaod_cpu;
namespace sqint = sqaod_internal;

template<class real>
void sqcpu::CPUDenseGraphFormulas<real>::calculate_E(real *E, const Matrix &W, const Vector &x) {
    sqint::matrixCheckIfSymmetric(W, __func__);
    Formulas::calculate_E(E, W, x);
}

template<class real>
void sqcpu::CPUDenseGraphFormulas<real>::calculate_E(Vector *E, const Matrix &W, const Matrix &x) {
    sqint::matrixCheckIfSymmetric(W, __func__);
    Formulas::calculate_E(E, W, x);
}

template<class real> void sqcpu::CPUDenseGraphFormulas<real>::
calculateHamiltonian(Vector *h, Matrix *J, real *c, const Matrix &W) {
    sqint::matrixCheckIfSymmetric(W, __func__);
    Formulas::calculateHamiltonian(h, J, c, W);
}

template<class real> void sqcpu::CPUDenseGraphFormulas<real>::
calculate_E(real *E, const Vector &h, const Matrix &J, real c, const Vector &q) {
    sqint::matrixCheckIfSymmetric(J, __func__);
    Formulas::calculate_E(E, h, J, c, q);
}

template<class real> void sqcpu::CPUDenseGraphFormulas<real>::
calculate_E(Vector *E, const Vector &h, const Matrix &J, real c, const Matrix &q) {
    sqint::matrixCheckIfSymmetric(J, __func__);
    Formulas::calculate_E(E, h, J, c, q);
}


template<class real> void sqcpu::CPUBipartiteGraphFormulas<real>::
calculate_E(real *E,
            const Vector &b0, const Vector &b1, const Matrix &W,
            const Vector &x0, const Vector &x1) {
    Formulas::calculate_E(E, b0, b1, W, x0, x1);
}

template<class real> void sqcpu::CPUBipartiteGraphFormulas<real>::
calculate_E(Vector *E,
            const Vector &b0, const Vector &b1, const Matrix &W,
            const Matrix &x0, const Matrix &x1) {
    Formulas::calculate_E(E, b0, b1, W, x0, x1);
}

template<class real> void sqcpu::CPUBipartiteGraphFormulas<real>::
calculate_E_2d(Matrix *E,
               const Vector &b0, const Vector &b1, const Matrix &W,
               const Matrix &x0, const Matrix &x1) {
    Formulas::calculate_E_2d(E, b0, b1, W, x0, x1);
}

template<class real> void sqcpu::CPUBipartiteGraphFormulas<real>::
calculateHamiltonian(Vector *h0, Vector *h1, Matrix *J, real *c,
                     const Vector &b0, const Vector &b1, const Matrix &W) {
    Formulas::calculateHamiltonian(h0, h1, J, c, b0, b1, W);
}

template<class real> void sqcpu::CPUBipartiteGraphFormulas<real>::
calculate_E(real *E,
            const Vector &h0, const Vector &h1, const Matrix &J, real c,
            const Vector &q0, const Vector &q1) {
    Formulas::calculate_E(E, h0, h1, J, c, q0, q1);
}

template<class real> void sqcpu::CPUBipartiteGraphFormulas<real>::
calculate_E(Vector *E,
            const Vector &h0, const Vector &h1, const Matrix &J, real c,
            const Matrix &q0, const Matrix &q1) {
    Formulas::calculate_E(E, h0, h1, J, c, q0, q1);
}


template struct sqcpu::CPUDenseGraphFormulas<double>;
template struct sqcpu::CPUDenseGraphFormulas<float>;

template struct sqcpu::CPUBipartiteGraphFormulas<double>;
template struct sqcpu::CPUBipartiteGraphFormulas<float>;

