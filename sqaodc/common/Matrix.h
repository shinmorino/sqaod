#pragma once

#include <sqaodc/common/defines.h>
#include <sqaodc/common/types.h>
#include <sqaodc/common/Array.h>
#include <sqaodc/common/os_dependent.h>

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

    const MatrixType<V> &operator=(const V &v);

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
        copy_data(this, src);
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
            aligned_free(data);
        data = nullptr;
    }
    
    void resize(SizeType _rows, SizeType _cols) {
        assert(!mapped); /* mapping state not allowed */
        if ((_rows != rows) || (_cols != cols)) {
            aligned_free(data);
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

    V sum() const;

    V min() const;
    
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

private:
    static
    void copy_data(MatrixType<V> *dst, const MatrixType<V> &src);

};

/* Matrix operator */
template<class V>
bool operator==(const MatrixType<V> &lhs, const MatrixType<V> &rhs);

template<class V>
bool operator!=(const MatrixType<V> &lhs, const MatrixType<V> &rhs);

template<class V>
const MatrixType<V> &operator*=(MatrixType<V> &mat, const V &v);

template<class newV, class V>
sqaod::MatrixType<newV> cast(const MatrixType<V> &mat);



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

    const VectorType<V> &operator=(const V &v);

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
            aligned_free(data);
        data = nullptr;
    }
    
    void resize(SizeType _size) {
        assert(!mapped);
        if (_size != size) {
            aligned_free(data);
            allocate(_size);
        }
    }

    V &operator()(IdxType idx) {
        return data[idx];
    }
    
    const V &operator()(IdxType idx) const {
        return data[idx];
    }

    V sum() const;

    V min() const;
    
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
bool operator==(const VectorType<V> &lhs, const VectorType<V> &rhs);

template<class V>
bool operator!=(const VectorType<V> &lhs, const VectorType<V> &rhs);

template<class V>
const VectorType<V> &operator*=(VectorType<V> &vec, const V &v);

/* cast */
template<class newV, class V>
sqaod::VectorType<newV> cast(const VectorType<V> &vec);



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
