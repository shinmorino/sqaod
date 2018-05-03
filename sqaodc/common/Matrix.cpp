#include "Matrix.h"
#include "EigenBridge.h"

using namespace sqaod;

template<class V>
MatrixType<V> MatrixType<V>::eye(SizeType dim) {
    Matrix mat = zeros(dim, dim);
    for (SizeType idx = 0; idx < dim; ++idx)
        mat(idx, idx) = V(1.);
    return mat;
}

template<class V>
MatrixType<V> MatrixType<V>::zeros(SizeType rows, SizeType cols) {
    Matrix mat(rows, cols);
    mapTo(mat) = EigenMatrixType<V>::Zero(rows, cols);
    return mat;
}

template<class V>
MatrixType<V> MatrixType<V>::ones(SizeType rows, SizeType cols) {
    MatrixType<V> mat(rows, cols);
    mapTo(mat) = EigenMatrixType<V>::Ones(rows, cols);
    return mat;
}

template<class V>
const MatrixType<V> &MatrixType<V>::operator=(const V &v) {
    mapTo(*this).array() = v;
    return *this;
}

template<class V>
V MatrixType<V>::sum() const {
    return mapTo(*this).sum();
}

template<class V>
V MatrixType<V>::min() const {
    return mapTo(*this).minCoeff();
}

/* Matrix operator */
template<class V>
bool sqaod::operator==(const MatrixType<V> &lhs, const MatrixType<V> &rhs) {
    return mapTo(lhs) == mapTo(rhs);
}

template<class V>
bool sqaod::operator!=(const MatrixType<V> &lhs, const MatrixType<V> &rhs) {
    return mapTo(lhs) != mapTo(rhs);
}

template<class V>
const MatrixType<V> &sqaod::operator*=(MatrixType<V> &mat, const V &v) {
    mapTo(mat) *= v;
    return mat;
}

template<class newV, class V>
sqaod::MatrixType<newV> sqaod::cast(const MatrixType<V> &mat) {
    MatrixType<newV> newMat(mat.dim());
    mapTo(newMat) = mapTo(mat).template cast<newV>();
    return newMat;
}

template<class V>
void MatrixType<V>::copy_data(MatrixType<V> *dst, const MatrixType<V> &src) {
    mapTo(*dst) = mapTo(src);
}


template struct sqaod::MatrixType<float>;
template struct sqaod::MatrixType<double>;

template bool sqaod::operator==(const MatrixType<double> &, const MatrixType<double> &);
template bool sqaod::operator==(const MatrixType<float> &, const MatrixType<float> &);
template bool sqaod::operator==(const MatrixType<char> &, const MatrixType<char> &);

template bool sqaod::operator!=(const MatrixType<double> &, const MatrixType<double> &);
template bool sqaod::operator!=(const MatrixType<float> &, const MatrixType<float> &);
template bool sqaod::operator!=(const MatrixType<char> &, const MatrixType<char> &);

template const MatrixType<double> &sqaod::operator*=(MatrixType<double> &, const double &);
template const MatrixType<float> &sqaod::operator*=(MatrixType<float> &, const float &);
template const MatrixType<char> &sqaod::operator*=(MatrixType<char> &, const char &);

template sqaod::MatrixType<char> sqaod::cast(const MatrixType<double> &);
template sqaod::MatrixType<char> sqaod::cast(const MatrixType<float> &);
template sqaod::MatrixType<double> sqaod::cast(const MatrixType<char> &);
template sqaod::MatrixType<float> sqaod::cast(const MatrixType<char> &);
template sqaod::MatrixType<double> sqaod::cast(const MatrixType<float> &);
template sqaod::MatrixType<float> sqaod::cast(const MatrixType<double> &);
template sqaod::MatrixType<double> sqaod::cast(const MatrixType<double> &);
template sqaod::MatrixType<float> sqaod::cast(const MatrixType<float> &);


template<class V>
VectorType<V> VectorType<V>::zeros(SizeType size) {
    VectorType<V> vec(size);
    mapToRowVector(vec) = EigenRowVectorType<V>::Zero(size);
    return vec;
}

template<class V>
VectorType<V> VectorType<V>::ones(SizeType size) {
    VectorType<V> vec(size);
    sqaod::mapToRowVector(vec) = EigenRowVectorType<V>::Ones(size);
    return vec;
}

template<class V>
const VectorType<V> &VectorType<V>::operator=(const V &v) {
    mapToRowVector(*this).array() = v;
    return *this;
}

template<class V>
V VectorType<V>::sum() const {
    return mapToRowVector(*this).sum();
}

template<class V>
V VectorType<V>::min() const {
    return mapToRowVector(*this).minCoeff();
}

template<class V>
bool sqaod::operator==(const VectorType<V> &lhs, const VectorType<V> &rhs) {
    return mapToRowVector(lhs) == mapToRowVector(rhs);
}

template<class V>
bool sqaod::operator!=(const VectorType<V> &lhs, const VectorType<V> &rhs) {
    return mapToRowVector(lhs) != mapToRowVector(rhs);
}

template<class V>
const VectorType<V> &sqaod::operator*=(VectorType<V> &vec, const V &v) {
    mapToRowVector(vec) *= v;
    return vec;
}

template<class newV, class V>
sqaod::VectorType<newV> sqaod::cast(const VectorType<V> &vec) {
    VectorType<newV> newVec(vec.size);
    mapToRowVector(newVec) = mapToRowVector(vec).template cast<newV>();
    return newVec;
}



template struct sqaod::VectorType<float>;
template struct sqaod::VectorType<double>;

template bool sqaod::operator==(const VectorType<double> &, const VectorType<double> &);
template bool sqaod::operator==(const VectorType<float> &, const VectorType<float> &);
template bool sqaod::operator==(const VectorType<char> &, const VectorType<char> &);

template bool sqaod::operator!=(const VectorType<double> &, const VectorType<double> &);
template bool sqaod::operator!=(const VectorType<float> &, const VectorType<float> &);
template bool sqaod::operator!=(const VectorType<char> &, const VectorType<char> &);

template const VectorType<double> &sqaod::operator*=(VectorType<double> &, const double &);
template const VectorType<float> &sqaod::operator*=(VectorType<float> &, const float &);
template const VectorType<char> &sqaod::operator*=(VectorType<char> &, const char &);

template sqaod::VectorType<char> sqaod::cast(const VectorType<double> &);
template sqaod::VectorType<char> sqaod::cast(const VectorType<float> &);
template sqaod::VectorType<double> sqaod::cast(const VectorType<char> &);
template sqaod::VectorType<float> sqaod::cast(const VectorType<char> &);
template sqaod::VectorType<double> sqaod::cast(const VectorType<float> &);
template sqaod::VectorType<float> sqaod::cast(const VectorType<double> &);
template sqaod::VectorType<double> sqaod::cast(const VectorType<double> &);
template sqaod::VectorType<float> sqaod::cast(const VectorType<float> &);
