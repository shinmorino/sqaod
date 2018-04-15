#pragma once

#include <sqaodc/common/defines.h>
#include <sqaodc/common/types.h>
#include <sqaodc/cuda/DeviceObject.h>

namespace sqaod_cuda {

namespace sq = sqaod;

/* light-weight matrix classes for C++ API */

/* FIXME: Support stride and tile, memory alignment of 4-way vector. */

template<class V>
struct DeviceMatrixType : DeviceObject {
    typedef V ValueType;
    typedef sq::SizeType SizeType;
    
    DeviceMatrixType() {
        d_data = NULL;
        rows = cols = -1;
    }
    
    DeviceMatrixType(V *_d_data, SizeType _rows, SizeType _cols) {
        d_data = _d_data;
        rows = _rows;
        cols = _cols;
    }
    
    virtual ~DeviceMatrixType() {
    }

    sq::Dim dim() const { return sq::Dim(rows, cols); }

    V *row(sq::IdxType row) {
        return &d_data[row * cols];
    }
    const V *row(sq::IdxType row) const {
        return &d_data[row * cols];
    }

    V &operator()(sq::IdxType r, sq::IdxType c) {
        assert((0 <= r) && (r < (sq::IdxType)rows));
        assert((0 <= c) && (c < (sq::IdxType)cols));
        return d_data[r * cols + c];
    }
    
    const V &operator()(sq::IdxType r, sq::IdxType c) const {
        assert((0 <= r) && (r < (sq::IdxType)rows));
        assert((0 <= c) && (c < (sq::IdxType)cols));
        return d_data[r * cols + c];
    }

    SizeType rows, cols;
    V *d_data;

private:
    DeviceMatrixType(const DeviceMatrixType<V>&);
    virtual void get_data(void **ppv) { 
        *ppv = d_data;
        d_data = NULL;
        rows = cols = -1;
    }
};
    

template<class V>
struct DeviceVectorType : DeviceObject {
    typedef V ValueType;
    typedef sq::SizeType SizeType;
    
    DeviceVectorType() {
        d_data = NULL;
        size = -1;
    }

    DeviceVectorType(V *_d_data, SizeType _size) {
        d_data = _d_data;
        size = _size;
    }
    
    V &operator()(sq::IdxType idx) {
        return d_data[idx];
    }
    
    const V &operator()(sq::IdxType idx) const {
        return d_data[idx];
    }

    virtual ~DeviceVectorType() {
    }
    
    SizeType size;
    V *d_data;

private:
    DeviceVectorType(const DeviceVectorType<V>&);
    virtual void get_data(void **ppv) { 
        *ppv = d_data;
        d_data = NULL;
        size = -1;
    }
};



template<class V>
struct DeviceScalarType : DeviceObject {
    typedef V ValueType;

    DeviceScalarType() {
        d_data = NULL;
    }

    DeviceScalarType(V *_d_data) {
        d_data = _d_data;
    }
    
    virtual ~DeviceScalarType() {
    }
    
    V *d_data;
private:
    DeviceScalarType(const DeviceScalarType<V>&);
    virtual void get_data(void **ppv) { 
        *ppv = d_data;
        d_data = NULL;
    }
};

typedef DeviceVectorType<char> DeviceBitSet;
typedef DeviceMatrixType<char> DeviceBitMatrix;

}
