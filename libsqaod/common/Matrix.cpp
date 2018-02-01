#include "Matrix.h"

#include "EigenBridge.h"

using namespace sqaod;

template<class real>
MatrixType<real> MatrixType<real>::eye(SizeType dim) {
    Matrix mat = zeros(dim, dim);
    for (SizeType idx = 0; idx < dim; ++idx)
        mat(idx, idx) = real(1.);
    return mat;
}

template<class real>
MatrixType<real> MatrixType<real>::zeros(SizeType rows, SizeType cols) {
    Matrix mat(rows, cols);
    mapTo(mat) = EigenMatrixType<real>::Zero(rows, cols);
    return mat;
}

template<class real>
MatrixType<real> MatrixType<real>::ones(SizeType rows, SizeType cols) {
    MatrixType<real> mat(rows, cols);
    mapTo(mat) = EigenMatrixType<real>::Ones(rows, cols);
    return mat;
}

template struct sqaod::MatrixType<float>;
template struct sqaod::MatrixType<double>;


template<class real>
VectorType<real> VectorType<real>::zeros(SizeType size) {
    VectorType<real> vec(size);
    mapToRowVector(vec) = EigenRowVectorType<real>::Zero(size);
    return vec;
}

template<class real>
VectorType<real> VectorType<real>::ones(SizeType size) {
    VectorType<real> vec(size);
    sqaod::mapToRowVector(vec) = EigenRowVectorType<real>::Ones(size);
    return vec;
}

template struct sqaod::VectorType<float>;
template struct sqaod::VectorType<double>;

