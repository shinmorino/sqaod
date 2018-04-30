#pragma once

#include <sqaodc/common/Matrix.h>
#include <sqaodc/cuda/DeviceMatrix.h>
#include <sqaodc/cuda/DeviceArray.h>

namespace sqaod_cuda {

namespace sq = sqaod;

/* FIXME: memory store */

struct HostObjectAllocator {

    void *allocate(size_t size);

    void *allocate2d(sq::SizeType *stride, size_t width, size_t height);

    void deallocate(void *pv);

    template<class V>
    void allocate(V **v, size_t size);

    template<class V>
    void allocate2d(V **v, sq::SizeType *stride, size_t width, size_t height);

    /* pinned memory allocator for CPU objects */

    template<class V>
    void allocate(sq::MatrixType<V> *mat, sq::SizeType rows, sq::SizeType cols);

    template<class V>
    void allocate(sq::MatrixType<V> *mat, const sq::Dim &dim);

    template<class V>
    void allocate(sq::VectorType<V> *vec, sq::SizeType size);

    /* Pinned memory allocation for device objects */

    template<class V>
    void allocate(DeviceMatrixType<V> *mat, sq::SizeType rows, sq::SizeType cols);

    template<class V>
    void allocate(DeviceMatrixType<V> *mat, const sq::Dim &dim);

    template<class V>
    void allocate(DeviceVectorType<V> *vec, sq::SizeType size);

    template<class V>
    void allocate(DeviceScalarType<V> *sc);

    template<class V>
    void allocate(DeviceArrayType<V> *arr, sq::SizeType capacity);

    /* allocating host memory if data member is Null. */

    template<class V>
    void allocateIfNull(sq::MatrixType<V> *mat, const sq::Dim &dim);

    template<class V>
    void allocateIfNull(sq::VectorType<V> *vec, const sq::SizeType size);

    template<class V>
    void allocateIfNull(DeviceScalarType<V> *sc);

    template<class V>
    void allocateIfNull(DeviceArrayType<V> *arr, sq::SizeType capacity);
    
    template<class T>
    void deallocate(T &obj);
};

template<class V>
void HostObjectAllocator::allocate(V **v, size_t size) {
    *v = (V*)allocate(size);
}

template<class V>
void HostObjectAllocator::allocate2d(V **v, sq::SizeType *stride, size_t width, size_t height) {
    *v = (V*)allocate2d(stride, width * sizeof(V), height);
}


template<class V> inline
void HostObjectAllocator::allocate(sq::MatrixType<V> *mat, const sq::Dim &dim) {
    return allocate(mat, dim.rows, dim.cols);
}

template<class V> inline
void HostObjectAllocator::allocate(sq::MatrixType<V> *mat, sq::SizeType rows, sq::SizeType cols) {
    allocate(&mat->data, &mat->stride, rows, cols);
    mat->rows = rows;
    mat->cols = cols;
}

template<class V> inline
void HostObjectAllocator::allocate(sq::VectorType<V> *vec, sq::SizeType size) {
    allocate(&vec->data, size);
    vec->size = size;
}

template<class V> inline void HostObjectAllocator::
allocate(DeviceMatrixType<V> *mat, sq::SizeType rows, sq::SizeType cols) {
    allocate2d(&mat->d_data, &mat->stride, cols, rows);
    mat->rows = rows;
    mat->cols = cols;
}

template<class V> inline void HostObjectAllocator::
allocate(DeviceMatrixType<V> *mat, const sq::Dim &dim) {
    return allocate(mat, dim.rows, dim.cols);
}

template<class V> inline void HostObjectAllocator::
allocate(DeviceVectorType<V> *vec, sq::SizeType size) {
    vec->d_data = (V*)allocate(sizeof(V) * size);
    vec->size = size;
}

template<class V> inline
void HostObjectAllocator::allocate(DeviceScalarType<V> *sc) {
    sc->d_data = (V*)allocate(sizeof(V));
}

template<class V> inline
void HostObjectAllocator::allocate(DeviceArrayType<V> *arr, sq::SizeType capacity) {
    arr->d_data = (V*)allocate(sizeof(V) * capacity);
    arr->capacity = capacity;
    arr->size = 0;
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(sq::MatrixType<V> *mat, const sq::Dim &dim) {
    if (mat->data == NULL)
        allocate(mat, dim);
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(sq::VectorType<V> *vec, const sq::SizeType size) {
    if (vec->data == NULL)
        allocate(vec, size);
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(DeviceScalarType<V> *sc) {
    if (sc->data == NULL)
        allocate(sc);
}

template<class V> inline
void HostObjectAllocator::allocateIfNull(DeviceArrayType<V> *arr, sq::SizeType size) {
    if (arr->data == NULL)
        allocate(arr, size);
}

template<class T>
void HostObjectAllocator::deallocate(T &obj) {
    deallocate((void*)obj.d_data);
    obj.d_data = NULL;
}


}
