
#ifndef SQAOD_CUDA_CUDA_MATRIX_H__
#define SQAOD_CUDA_CUDA_MATRIX_H__

#include <common/Matrix.h>
#include <cuda/DeviceObject.h>

namespace sqaod_cuda {

/* light-weight matrix classes for C++ API */

template<class V>
struct DeviceMatrixType : DeviceObject {
    typedef V ValueType;
    typedef sqaod::SizeType SizeType;
    
    DeviceMatrixType() {
        d_data = NULL;
        rows = cols = (SizeType)-1;
    }
    
    DeviceMatrixType(V *_d_data, SizeType _rows, SizeType _cols) {
        d_data = _d_data;
        rows = _rows;
        cols = _cols;
    }
    
    virtual ~DeviceMatrixType() {
    }

    sqaod::Dim dim() const { return sqaod::Dim(rows, cols); }

    SizeType rows, cols;
    V *d_data;

private:
    DeviceMatrixType(const DeviceMatrixType<V>&);
    virtual void get_data(void **ppv) { 
        *ppv = d_data;
        d_data = NULL;
        rows = cols = (SizeType)-1;
    }
};
    

template<class V>
struct DeviceVectorType : DeviceObject {
    typedef V ValueType;
    typedef sqaod::SizeType SizeType;
    
    DeviceVectorType() {
        d_data = NULL;
        size = (SizeType)-1;
    }

    DeviceVectorType(V *_d_data, SizeType _size) {
        d_data = _d_data;
        size = _size;
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
        size = (SizeType)-1;
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

typedef DeviceVectorType<char> DeviceBits;
typedef DeviceMatrixType<char> DeviceBitMatrix;

}

        
#endif
