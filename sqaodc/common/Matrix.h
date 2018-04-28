#pragma once

#include <sqaodc/common/defines.h>
#include <sqaodc/common/types.h>
#include <sqaodc/common/UniformOp.h>
#include <sqaodc/common/Array.h>

namespace sqaod {

/* light-weight matrix classes for C++ API */

template<class V>
struct MatrixType {
    typedef V ValueType;
    typedef MatrixType<V> Matrix;
    
    explicit MatrixType() {
        resetState();
    }

    explicit MatrixType(SizeType _rows, SizeType _cols) {
        resetState();
        allocate(_rows, _cols);
    }

    explicit MatrixType(const Dim &dim) {
        resetState();
        allocate(dim.rows, dim.cols);
    }

    MatrixType(const MatrixType<V> &mat) {
        resetState();
        copyFrom(mat);
    }
    
    MatrixType(MatrixType<V> &&mat) noexcept {
        resetState();
        moveFrom(static_cast<MatrixType<V>&>(mat));
    }

    explicit MatrixType(V *_data, SizeType _rows, SizeType _cols, SizeType _stride) {
        data = _data;
        rows = _rows;
        cols = _cols;
        stride = _stride;
        mapped = true;
    }
    
    virtual ~MatrixType() {
        if (!mapped)
            free();
    }

    const MatrixType<V> &operator=(const MatrixType<V> &rhs) {
        copyFrom(rhs);
        return rhs;
    }

    void operator=(const V &v) {
        sqaod::fill(data, v, cols, rows, stride);
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
    
    void map(V *_data, SizeType _rows, SizeType _cols, SizeType _stride) {
        if (!mapped)
            free();
        mapped = true;
        data = _data;
        rows = _rows;
        cols = _cols;
        stride = _stride;
    }
    
    void copyFrom(const MatrixType<V> &src) {
        if (this == &src)
            return;
        if ((rows != src.rows) || (cols != src.cols)) {
            abortIf(mapped, "Unable to resize mapped matrix.");
            free();
        }
        if (data == nullptr)
            allocate(src.rows, src.cols);
        assert(stride == src.stride);
        for (IdxType row = 0; row < rows; ++row) {
            IdxType srcOffset = src.stride * row; 
            IdxType selfOffset = stride * row; 
            memcpy(&data[selfOffset], &src.data[srcOffset], sizeof(V) * cols);
        }
    }

    void moveFrom(MatrixType<V> &src) {
        if (this == &src)
            return;
        if (data != nullptr)
            free();
        /* updating this */
        rows = src.rows;
        cols = src.cols;
        stride = src.stride;
        data = src.data;
        mapped = src.mapped;
        /* clean up src */
        src.resetState();
    }
    
    void allocate(SizeType _rows, SizeType _cols) {
        assert(!mapped);
        rows = _rows;
        cols = _cols;
        const int SIMD_WORDS = SQAODC_SIMD_ALIGNMENT / sizeof(V);
        stride = roundUp(cols, SIMD_WORDS);
        data = (V*)aligned_alloc(SQAODC_SIMD_ALIGNMENT, rows * stride * sizeof(V));
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

    void resize(const Dim &dim) {
        resize(dim.rows, dim.cols);
    }

    V &operator()(IdxType r, IdxType c) {
        assert((0 <= r) && (r < (IdxType)rows));
        assert((0 <= c) && (c < (IdxType)cols));
        return data[r * stride + c];
    }
    
    const V &operator()(IdxType r, IdxType c) const {
        assert((0 <= r) && (r < (IdxType)rows));
        assert((0 <= c) && (c < (IdxType)cols));
        return data[r * stride + c];
    }

    V sum() const {
        return sqaod::sum(data, cols, rows, stride);
    }

    V min() const {
        return sqaod::min(data, cols, rows, stride);
    }
    
    SizeType rows, cols, stride;
    V *data;
    bool mapped;

    /* static initializers */
    static
    MatrixType<V> eye(SizeType dim);

    static
    MatrixType<V> zeros(SizeType rows, SizeType cols);

    static
    MatrixType<V> zeros(const Dim &dim) {
        return zeros(dim.rows, dim.cols);
    }

    static
    MatrixType<V> ones(SizeType rows, SizeType cols);

    static
    MatrixType<V> ones(const Dim &dim) {
        return ones(dim.rows, dim.cols);
    }
};

/* Matrix operator */
template<class V>
bool operator==(const MatrixType<V> &lhs, const MatrixType<V> &rhs) {
    if (lhs.dim() != rhs.dim())
        return false;
    for (IdxType row = 0; row < lhs.rows; ++row) {
        IdxType offset = lhs.stride * row;
        if (memcmp(&lhs.data[offset], &rhs.data[offset], sizeof(V) * lhs.cols) != 0)
            return false;
    }
    return true;
}

template<class V>
bool operator!=(const MatrixType<V> &lhs, const MatrixType<V> &rhs) {
    return !(lhs == rhs);
}

template<class V>
MatrixType<V> &operator*=(MatrixType<V> &mat, const V &v) {
    multiply(mat.data, v, mat.cols, mat.rows, mat.stride);
    return mat;
}

template<class newV, class V>
sqaod::MatrixType<newV> cast(const MatrixType<V> &mat) {
    MatrixType<newV> newMat(mat.dim());
    cast(newMat.data, mat.data, mat.cols, mat.rows, newMat.stride, mat.stride);
    return newMat;
}



template<class V>
struct VectorType {
    typedef V ValueType;

    explicit VectorType() {
        resetState();
    }

    explicit VectorType(SizeType _size) {
        resetState();
        allocate(_size);
    }

    VectorType(const VectorType<V> &vec) {
        resetState();
        copyFrom(vec);
    }
    
    VectorType(VectorType<V> &&vec) noexcept {
        resetState();
        moveFrom(static_cast<VectorType<V>&>(vec));
    }
    
    explicit VectorType(V *_data, SizeType _size) {
        data = _data;
        size = _size;
        mapped = true;
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

    void map(V *_data, SizeType _size) {
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

    void operator=(const V &v) {
        sqaod::fill(data, v, size);
    }

    const VectorType<V> &operator=(VectorType<V> &&rhs) noexcept {
        moveFrom(static_cast<VectorType<V>&>(rhs));
        return *this;
    }

    void copyFrom(const VectorType<V> &src) {
        if (this == &src)
            return;
        if (size != src.size) {
            abortIf(mapped, "Unable to resize mapped vector.");
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
        if ((!mapped) && (data != nullptr))
            free();
        size = src.size;
        data = src.data;
        mapped = false;
        src.size = -1;
        src.data = nullptr;
    }
    
    void allocate(SizeType _size) {
        assert(!mapped);
        size = _size;
        SizeType alignedSize = roundUp(size * sizeof(V), SQAODC_SIMD_ALIGNMENT);
        data = (V*)aligned_alloc(SQAODC_SIMD_ALIGNMENT, alignedSize);
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

    V sum() const {
        return sqaod::sum(data, size);
    }

    V min() const {
        return sqaod::min(data, size);
    }
    
    SizeType size;
    V *data;
    bool mapped;

    /* static initializers */
    static
    VectorType<V> zeros(SizeType size);

    static
    VectorType<V> ones(SizeType size);

};



/* Vector operator */

template<class V>
bool operator==(const VectorType<V> &lhs, const VectorType<V> &rhs) {
    if (lhs.size != rhs.size)
        return false;
    return memcmp(lhs.data, rhs.data, sizeof(V) * lhs.size) == 0;
}

template<class V>
bool operator!=(const VectorType<V> &lhs, const VectorType<V> &rhs) {
    return !(lhs == rhs);
}

template<class V>
VectorType<V> &operator*=(VectorType<V> &vec, const V &v) {
    multiply(vec.data, v, vec.size);
    return vec;
}

/* cast */
template<class newV, class V>
sqaod::VectorType<newV> cast(const VectorType<V> &vec) {
    VectorType<newV> newVec(vec.size);
    cast(newVec.data, vec.data, vec.size);
    return newVec;
}



typedef VectorType<char> BitSet;
typedef ArrayType<BitSet> BitSetArray;

/* BitSetPair */

struct BitSetPair {
    BitSetPair() { }
    BitSetPair(const BitSet &_bits0, const BitSet &_bits1)
            : bits0(_bits0), bits1(_bits1) { }
    BitSet bits0;
    BitSet bits1;
};

inline
bool operator==(const BitSetPair &lhs, const BitSetPair &rhs) {
    return (lhs.bits0 == rhs.bits0) && (lhs.bits1 == rhs.bits1);
}

inline
bool operator!=(const BitSetPair &lhs, const BitSetPair &rhs) {
    return !(lhs == rhs);
}


typedef ArrayType<BitSetPair> BitSetPairArray;

typedef MatrixType<char> BitMatrix;

//typedef VectorType<char> Spins;
//typedef std::vector<Bits> SpinsArray;
//typedef std::vector<std::pair<Bits, Bits> > SpinsPairArray;
//typedef VectorType<char> SpinsMatrix;
    
}
