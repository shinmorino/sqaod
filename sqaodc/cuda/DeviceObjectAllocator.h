#pragma once

#include <sqaodc/cuda/DeviceMatrix.h>
#include <sqaodc/cuda/DeviceMemoryStore.h>
#include <sqaodc/cuda/DeviceStream.h>
#include <sqaodc/cuda/DeviceArray.h>

namespace sqaod_cuda {

namespace sq = sqaod;

struct DeviceObjectAllocator {

    void *allocate(size_t size);

    void deallocate(void *pv);

    template<class V>
    void allocate(V **v, size_t size);

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
    
    template<class V>
    void allocateIfNull(DeviceMatrixType<V> *mat, const sq::Dim &dim);

    template<class V>
    void allocateIfNull(DeviceVectorType<V> *vec, const sq::SizeType size);

    template<class V>
    void allocateIfNull(DeviceScalarType<V> *sc);

    template<class V>
    void allocateIfNull(DeviceArrayType<V> *arr, sq::SizeType size);
    
    void deallocate(DeviceObject &obj);

    void set(DeviceMemoryStore *memStore) {
        memStore_ = memStore;
    }
    
private:
    DeviceMemoryStore *memStore_;
};

template<class V> inline
void DeviceObjectAllocator::allocate(V **v, size_t size) {
    *v = (V*)allocate(sizeof(V) * size);
}


template<class V> inline
void DeviceObjectAllocator::allocate(DeviceMatrixType<V> *mat, const sq::Dim &dim) {
    return allocate(mat, dim.rows, dim.cols);
}

template<class V> inline
void DeviceObjectAllocator::allocate(DeviceMatrixType<V> *mat, sq::SizeType rows, sq::SizeType cols) {
    mat->stride = sq::roundUp(cols, DeviceMatrixType<V>::STRIDE_ALIGNMENT);
    allocate(&mat->d_data, mat->stride * rows);
    mat->rows = rows;
    mat->cols = cols;
}

template<class V> inline
void DeviceObjectAllocator::allocate(DeviceVectorType<V> *vec, sq::SizeType size) {
    allocate(&vec->d_data, size);
    vec->size = size;
}

template<class V> inline
void DeviceObjectAllocator::allocate(DeviceScalarType<V> *sc) {
    allocate(&sc->d_data, 1);
}

template<class V> inline
void DeviceObjectAllocator::allocate(DeviceArrayType<V> *arr, sq::SizeType capacity) {
    allocate(&arr->d_data, capacity);
    arr->capacity = capacity;
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceMatrixType<V> *mat, const sq::Dim &dim) {
    if (mat->d_data == NULL)
        allocate(mat, dim);
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceVectorType<V> *vec, const sq::SizeType size) {
    if (vec->d_data == NULL)
        allocate(vec, size);
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceScalarType<V> *sc) {
    if (sc->d_data == NULL)
        allocate(sc);
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceArrayType<V> *arr, sq::SizeType size) {
    if (arr->d_data == NULL)
        allocate(arr, size);
}

}
