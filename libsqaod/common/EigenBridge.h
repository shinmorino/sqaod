#pragma once


#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable:4267)
#endif

#ifdef SQAOD_WITH_BLAS
#  define EIGEN_USE_BLAS
#endif

#include <Eigen/Core>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <common/Matrix.h>


namespace sqaod {


template<class real>
using EigenMatrixType = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<class real>
using EigenRowVectorType = Eigen::Matrix<real, 1, Eigen::Dynamic>;
template<class real>
using EigenColumnVectorType = Eigen::Matrix<real, Eigen::Dynamic, 1>;
template<class real>
using EigenMappedMatrixType = Eigen::Map<EigenMatrixType<real>>;
template<class real>
using EigenMappedRowVectorType = Eigen::Map<EigenRowVectorType<real>>;
template<class real>
using EigenMappedColumnVectorType = Eigen::Map<EigenColumnVectorType<real>>;

typedef EigenMatrixType<char> EigenBitMatrix;


/* Mapping matrix */

template<class V>
MatrixType<V> mapFrom(EigenMatrixType<V> &matrix) {
    return MatrixType<V>(matrix.data(), matrix.rows(), matrix.cols());
}

template<class V>
EigenMappedMatrixType<V> mapTo(MatrixType<V> &mat) {
    return EigenMappedMatrixType<V>(mat.data, mat.rows, mat.cols);
}

template<class V>
const EigenMappedMatrixType<V> mapTo(const MatrixType<V> &mat) {
    return EigenMappedMatrixType<V>(mat.data, mat.rows, mat.cols);
}


/* Mapping vector */

template<class V>
VectorType<V> mapAsVectorFrom(EigenMatrixType<V> &matrix) {
    assert((matrix.rows() == 1) || (matrix.cols() == 1));
    return VectorType<V>(matrix.data(), std::max(matrix.rows(), matrix.cols()));
}

template<class V>
VectorType<V> mapFrom(EigenRowVectorType<V> &vec) {
    assert(vec.rows() == 1);
    return VectorType<V>(vec.data(), vec.cols());
}

template<class V>
EigenMappedRowVectorType<V> mapToRowVector(VectorType<V> &vec) {
    return EigenMappedRowVectorType<V>(vec.data, 1, vec.size);
}

template<class V>
const EigenMappedRowVectorType<V> mapToRowVector(const VectorType<V> &vec) {
    return EigenMappedRowVectorType<V>(vec.data, 1, vec.size);
}

template<class V>
EigenMappedColumnVectorType<V> mapToColumnVector(VectorType<V> &vec) {
    return EigenMappedColumnVectorType<V>(vec.data, vec.size, 1);
}

template<class V>
const EigenMappedColumnVectorType<V> mapToColumnVector(const VectorType<V> &vec) {
    return EigenMappedColumnVectorType<V>(vec.data, vec.size, 1);
}

template<class newV, class V>
VectorType<newV> extractRow(const EigenMatrixType<V> &emat, IdxType rowIdx) {
    VectorType<newV> vec(emat.cols());
    for (IdxType idx = 0; idx < (IdxType)vec.size; ++idx)
        vec(idx) = (newV)emat(rowIdx, idx);
    return vec;
}

template<class newV, class V>
VectorType<newV> extractColumn(const EigenMatrixType<V> &emat, IdxType colIdx) {
    VectorType<newV> vec(emat.rows());
    for (IdxType idx = 0; idx < (IdxType)vec.size; ++idx)
        vec(idx) = (newV)emat(idx, colIdx);
    return vec;
}


}
