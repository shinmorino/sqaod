#include "Matrix.h"

using namespace sqaod;

template<class real>
MatrixType<real> MatrixType<real>::eye(SizeType dim) {
    MatrixType<real> mat(dim, dim);
    EigenMappedMatrix mapped = mat.map();
    mapped = EigenMatrix::Zero(dim, dim);
    for (SizeType idx = 0; idx < dim; ++idx) {
        mapped(idx, idx) = real(1.);
    }
    return mat;
}

template<class real>
MatrixType<real> MatrixType<real>::zeros(SizeType rows, SizeType cols) {
    MatrixType<real> mat(rows, cols);
    mat.map() = EigenMatrix::Zero(rows, cols);
    return mat;
}

template struct sqaod::MatrixType<float>;
template struct sqaod::MatrixType<double>;

