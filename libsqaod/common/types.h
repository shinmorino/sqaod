#pragma once

/* Simple/primitive types widely used in sqaod */

#include <stddef.h>


namespace sqaod {

typedef unsigned int SizeType;
typedef int IdxType;

typedef unsigned long long PackedBits;

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
