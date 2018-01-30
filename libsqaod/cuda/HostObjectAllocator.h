#pragma once

#include <common/Matrix.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>

namespace sqaod_cuda {

struct HostObjectAllocator {

    void *allocate(size_t size);

    void deallocate(void *pv);

    template<class V>
    void allocate(sqaod::MatrixType<V> *mat, sqaod::SizeType rows, sqaod::SizeType cols);

    template<class V>
    void allocate(sqaod::MatrixType<V> *mat, const sqaod::Dim &dim);

    template<class V>
    void allocate(sqaod::VectorType<V> *vec, sqaod::SizeType size);

    template<class V>
    void allocate(DeviceScalarType<V> *sc);

    template<class V>
    void allocate(DeviceArrayType<V> *arr, sqaod::SizeType capacity);
    
    template<class V>
    void allocateIfNull(sqaod::MatrixType<V> *mat, const sqaod::Dim &dim);

    template<class V>
    void allocateIfNull(sqaod::VectorType<V> *vec, const sqaod::SizeType size);

    template<class V>
    void allocateIfNull(DeviceScalarType<V> *sc);

    template<class V>
    void allocateIfNull(DeviceArrayType<V> *arr, sqaod::SizeType capacity);
    
    template<class T>
    void deallocate(T &obj);
};


template<class V> inline
void HostObjectAllocator::allocate(sqaod::MatrixType<V> *mat, const sqaod::Dim &dim) {
    return allocate(mat, dim.rows, dim.cols);
}

template<class V> inline
void HostObjectAllocator::allocate(sqaod::MatrixType<V> *mat, sqaod::SizeType rows, sqaod::SizeType cols) {
    mat->data = allocate(sizeof(V) * rows * cols);
    mat->rows = rows;
    mat->cols = cols;
}

template<class V> inline
void HostObjectAllocator::allocate(sqaod::VectorType<V> *vec, sqaod::SizeType size) {
    vec->data = (V*)allocate(sizeof(V) * size);
    vec->size = size;
}

template<class V> inline
void HostObjectAllocator::allocate(DeviceScalarType<V> *sc) {
    sc->d_data = (V*)allocate(sizeof(V));
}

template<class V> inline
void HostObjectAllocator::allocate(DeviceArrayType<V> *arr, sqaod::SizeType capacity) {
    arr->d_data = (V*)allocate(sizeof(V) * capacity);
    arr->capacity = capacity;
    arr->size = 0;
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(sqaod::MatrixType<V> *mat, const sqaod::Dim &dim) {
    if (mat->data == NULL)
        allocate(mat, dim);
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(sqaod::VectorType<V> *vec, const sqaod::SizeType size) {
    if (vec->data == NULL)
        allocate(vec, size);
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(DeviceScalarType<V> *sc) {
    if (sc->data == NULL)
        allocate(sc);
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(DeviceArrayType<V> *arr, sqaod::SizeType size) {
    if (arr->data == NULL)
        allocate(arr, size);
}

template<class T>
void HostObjectAllocator::deallocate(T &obj) {
    deallocate((void*)obj.d_data);
    obj.d_data = NULL;
}


}
