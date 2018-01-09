#ifndef COMMON_MATRIX_H__
#define COMMON_MATRIX_H__

#define EIGEN_NO_CUDA 1
#ifdef __CUDACC__
#  undef __CUDACC__
#endif
#ifdef __CUDACC_VER__
#  undef __CUDACC_VER__
#endif

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable:4267)
#endif

#include <Eigen/Core>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <common/Array.h>

namespace sqaod {


#define THROW_IF(cond, msg) if (cond) throw std::runtime_error(msg);
#define THROW(msg) throw std::runtime_error(msg);

    
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
    
/* light-weight matrix classes for C++ API */
    
struct Dim {
    Dim(SizeType _rows, SizeType _cols) {
        rows = _rows;
        cols = _cols;
    }
    SizeType rows, cols;
};

template<class V>
struct MatrixType {
    typedef V ValueType;
    typedef EigenMatrixType<V> EigenMatrix;
    typedef EigenMappedMatrixType<V> EigenMappedMatrix;
    
    MatrixType() {
        resetState();
    }

    MatrixType(SizeType _rows, SizeType _cols) {
        resetState();
        allocate(_rows, _cols);
    }

    MatrixType(const Dim &dim) {
        resetState();
        allocate(dim.rows, dim.cols);
    }

    /* mapping */
    MatrixType(EigenMatrix &matrix) {
        rows = matrix.rows();
        cols = matrix.cols();
        data = matrix.data();
        mapped = true;
    }

    MatrixType(const MatrixType<V> &mat) {
        resetState();
        copyFrom(mat);
    }
    
    MatrixType(MatrixType<V> &&mat) noexcept {
        resetState();
        moveFrom(static_cast<MatrixType<V>&>(mat));
    }
    
    virtual ~MatrixType() {
        if (!mapped)
            free();
    }

    const MatrixType<V> &operator=(const MatrixType<V> &rhs) {
        copyFrom(rhs);
        return rhs;
    }

    const MatrixType<V> &operator=(MatrixType<V> &&rhs) noexcept {
        moveFrom(static_cast<MatrixType<V>&>(rhs));
        return *this;
    }

    Dim dim() const { return Dim(rows, cols); }

    void resetState() {
        data = nullptr;
        rows = cols = -1;
        mapped = false;
    }
    
    void set(V *_data, SizeType _rows, SizeType _cols) {
        if (!mapped)
            free();
        mapped = true;
        data = _data;
        rows = _rows;
        cols = _cols;
    }
    
    void copyFrom(const MatrixType<V> &src) {
        if (this == &src)
            return;
        if ((rows != src.rows) || (cols != src.cols)) {
            THROW_IF(mapped, "Unable to resize mapped matrix.");
            free();
        }
        if (data == nullptr)
            allocate(src.rows, src.cols);
        memcpy(data, src.data, sizeof(V) * rows * cols);
    }

    void moveFrom(MatrixType<V> &src) {
        if (this == &src)
            return;
        if (src.mapped) {
            copyFrom(src);
            return;
        }
        if (data != nullptr)
            free();
        /* updating this */
        rows = src.rows;
        cols = src.cols;
        data = src.data;
        mapped = false;
        /* clean up src */
        src.rows = src.cols = -1;
        src.data = nullptr;
    }
    
    void allocate(SizeType _rows, SizeType _cols) {
        assert(!mapped);
        rows = _rows;
        cols = _cols;
        data = (V*)malloc(rows * cols * sizeof(V));
    }
    
    void free() {
        assert(!mapped);
        rows = cols = -1;
        if (data != nullptr)
            ::free(data);
        data = nullptr;
    }
    
    void resize(SizeType _rows, SizeType _cols) {
        assert(!mapped); /* mapping state not allowed */
        if ((_rows != rows) || (_cols != cols)) {
            ::free(data);
            allocate(_rows, _cols);
        }
    }

    V &operator()(IdxType r, IdxType c) {
        return data[r * cols + c];
    }
    
    const V &operator()(IdxType r, IdxType c) const {
        return data[r * cols + c];
    }
    
    EigenMappedMatrix map() {
        return EigenMappedMatrix(data, rows, cols);
    }

    const EigenMappedMatrix map() const {
        return EigenMappedMatrix(data, rows, cols);
    }

    template<class newV>
    sqaod::MatrixType<newV> cast() const {
        MatrixType<newV> newMat(dim());
        newMat.map() = map().cast<newV>();
        return newMat;
    }
    
    SizeType rows, cols;
    V *data;
    bool mapped;
};
    

template<class V>
struct VectorType {
    typedef V ValueType;
    typedef EigenMatrixType<V> EigenMatrix;
    typedef EigenRowVectorType<V> EigenRowVector;
    typedef EigenColumnVectorType<V> EigenColumnVector;
    typedef EigenMappedMatrixType<V> EigenMappedMatrix;
    typedef EigenMappedRowVectorType<V> EigenMappedRowVector;
    typedef EigenMappedColumnVectorType<V> EigenMappedColumnVector;

    VectorType() {
        resetState();
    }

    VectorType(SizeType _size) {
        resetState();
        allocate(_size);
    }

    VectorType(EigenMatrix &matrix) {
        assert((matrix.rows() == 1) || (matrix.cols() == 1));
        size = std::max(matrix.rows(), matrix.cols());
        data = matrix.data();
        mapped = true;
    }
    
    VectorType(const EigenMatrix &matrix) {
        assert((matrix.rows() == 1) || (matrix.cols() == 1));
        resetState();
        allocate(std::max(matrix.rows(), matrix.cols()));
        memcpy(data, matrix.data(), sizeof(V) * size);
    }
    
    VectorType(EigenRowVector &vec) {
        size = vec.cols();
        data = vec.data();
        mapped = true;
    }

    VectorType(const VectorType<V> &vec) {
        resetState();
        copyFrom(vec);
    }
    
    VectorType(VectorType<V> &&vec) noexcept {
        resetState();
        moveFrom(static_cast<VectorType<V>&>(vec));
    }
    
    virtual ~VectorType() {
        if (!mapped)
            free();
    }

    void resetState() {
        data = nullptr;
        size = -1;
        mapped = false;
    }
    
    void set(V *_data, SizeType _size) {
        if (!mapped)
            free();
        mapped = true;
        data = _data;
        size = _size;
    }
    
    const VectorType<V> &operator=(const VectorType<V> &rhs) {
        copyFrom(rhs);
        return rhs;
    }

    const VectorType<V> &operator=(VectorType<V> &&rhs) noexcept {
        moveFrom(static_cast<VectorType<V>&>(rhs));
        return *this;
    }

    void copyFrom(const VectorType<V> &src) {
        if (this == &src)
            return;
        if (size != src.size) {
            THROW_IF(mapped, "Unable to resize mapped vector.");
            free();
        }
        if (data == nullptr)
            allocate(src.size);
        memcpy(data, src.data, sizeof(V) * size);
    }

    void moveFrom(VectorType<V> &src) {
        if (this == &src)
            return;
        if (src.mapped) {
            copyFrom(src);
            return;
        }
        if (data != nullptr)
            free();
        size = src.size;
        data = src.data;
        mapped = false;
        src.size = -1;
        src.data = nullptr;
    }
    
    void allocate(SizeType _size) {
        assert(!mapped);
        /* FIXME: alligned mem allocator */
        size = _size;
        data = (V*)malloc(size * sizeof(V));
    }
    
    void free() {
        assert(!mapped);
        size = -1;
        if (data != nullptr)
            ::free(data);
        data = nullptr;
    }
    
    void resize(SizeType _size) {
        assert(!mapped);
        if (_size != size) {
            ::free(data);
            allocate(_size);
        }
    }

    V &operator()(IdxType idx) {
        return data[idx];
    }
    
    const V &operator()(IdxType idx) const {
        return data[idx];
    }
    
    EigenMappedRowVector mapToRowVector() {
        return EigenMappedRowVector(data, 1, size);
    }

    const EigenMappedRowVector mapToRowVector() const {
        return EigenMappedRowVector(data, 1, size);
    }
    
    EigenMappedColumnVector mapToColumnVector() {
        return EigenMappedColumnVector(data, size, 1);
    }

    const EigenMappedColumnVector mapToColumnVector() const {
        return EigenMappedColumnVector(data, size, 1);
    }

    template<class newV>
    sqaod::VectorType<newV> cast() const {
        sqaod::VectorType<newV> newVec(size);
        // sqaod::EigenMappedRowVectorType<real> ev = vec.mapToRowVector();
        newVec.mapToRowVector() = mapToRowVector().cast<newV>();;
        return newVec;
    }
    
    SizeType size;
    V *data;
    bool mapped;
};
    

typedef VectorType<char> Bits;
typedef MatrixType<char> BitMatrix;
typedef ArrayType<Bits> BitsArray;
typedef ArrayType<std::pair<Bits, Bits> > BitsPairArray;

//typedef VectorType<char> Spins;
//typedef std::vector<Bits> SpinsArray;
//typedef std::vector<std::pair<Bits, Bits> > SpinsPairArray;
//typedef VectorType<char> SpinsMatrix;
    
}

#endif
