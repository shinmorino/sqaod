#pragma once

/* Simple/primitive types widely used in sqaod */

#include <stddef.h>


namespace sqaod {

typedef unsigned int SizeType;
typedef int IdxType;

typedef unsigned long long PackedBits;

#ifdef __NVCC__
#  define ALLOW_DEVICE_CALL __host__ __device__ __forceinline__
#else
#  define ALLOW_DEVICE_CALL
#endif

struct PackedBitsPair {
    ALLOW_DEVICE_CALL PackedBitsPair() { }
    ALLOW_DEVICE_CALL PackedBitsPair(PackedBits _bits0, PackedBits _bits1)
            : bits0(_bits0), bits1(_bits1) { }
    PackedBits bits0;
    PackedBits bits1;
};

#undef ALLOW_DEVICE_CALL

struct Dim {
    Dim() {
        rows = cols == (SizeType)-1;
    }
    Dim(SizeType _rows, SizeType _cols) {
        rows = _rows;
        cols = _cols;
    }

    Dim transpose() const {
        return Dim(cols, rows);
    }

    SizeType rows, cols;

    friend bool operator==(const Dim &lhs, const Dim &rhs) {
        return (lhs.rows == rhs.rows) && (lhs.rows == rhs.rows);
    }
    friend bool operator!=(const Dim &lhs, const Dim &rhs) {
        return !(lhs == rhs);
    }
    
};

}
