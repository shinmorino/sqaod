#ifndef SQAOD_CUDA_DEVICEOBJECTALLOCATOR_H__
#define SQAOD_CUDA_DEVICEOBJECTALLOCATOR_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceMemoryStore.h>
#include <cuda/DeviceStream.h>
#include <cuda/DeviceArray.h>

namespace sqaod_cuda {

struct DeviceObjectAllocator {

    void *allocate(size_t size);

    void deallocate(void *pv);

    template<class V>
    void allocate(DeviceMatrixType<V> *mat, sqaod::SizeType rows, sqaod::SizeType cols);

    template<class V>
    void allocate(DeviceMatrixType<V> *mat, const sqaod::Dim &dim);

    template<class V>
    void allocate(DeviceVectorType<V> *vec, sqaod::SizeType size);

    template<class V>
    void allocate(DeviceScalarType<V> *sc);

    template<class V>
    void allocate(DeviceArrayType<V> *arr, sqaod::SizeType capacity);
    
    template<class V>
    void allocateIfNull(DeviceMatrixType<V> *mat, const sqaod::Dim &dim);

    template<class V>
    void allocateIfNull(DeviceVectorType<V> *vec, const sqaod::SizeType size);

    template<class V>
    void allocateIfNull(DeviceScalarType<V> *sc);

    template<class V>
    void allocateIfNull(DeviceArrayType<V> *arr, sqaod::SizeType size);
    
    void deallocate(DeviceObject &obj);

    void set(DeviceMemoryStore *memStore) {
        memStore_ = memStore;
    }
    
private:
    DeviceMemoryStore *memStore_;
};

template<class V>
void DeviceObjectAllocator::allocate(V **v, size_t size) {
    *v = (V*)allocate(sizeof(V) * size);
}

template<class V> inline
void DeviceObjectAllocator::allocate(DeviceMatrixType<V> *mat, const sqaod::Dim &dim) {
    return allocate(mat, dim.rows, dim.cols);
}

template<class V>
void DeviceObjectAllocator::allocate(DeviceMatrixType<V> *mat, sqaod::SizeType rows, sqaod::SizeType cols) {
    mat->d_data = (V*)allocate(sizeof(V) * rows * cols);
    mat->rows = rows;
    mat->cols = cols;
}

template<class V>
void DeviceObjectAllocator::allocate(DeviceVectorType<V> *vec, sqaod::SizeType size) {
    vec->d_data = (V*)allocate(sizeof(V) * size);
    vec->size = size;
}

template<class V>
void DeviceObjectAllocator::allocate(DeviceScalarType<V> *sc) {
    sc->d_data = (V*)allocate(sizeof(V));
}

template<class V>
void DeviceObjectAllocator::allocate(DeviceArrayType<V> *arr, sqaod::SizeType capacity) {
    arr->d_data = (V*)allocate(sizeof(V) * capacity);
    arr->capacity = capacity;
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceMatrixType<V> *mat, const sqaod::Dim &dim) {
    if (mat->d_data == NULL)
        allocate(mat, dim);
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceVectorType<V> *vec, const sqaod::SizeType size) {
    if (vec->d_data == NULL)
        allocate(vec, size);
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceScalarType<V> *sc) {
    if (sc->d_data == NULL)
        allocate(sc);
}

template<class V> inline
void DeviceObjectAllocator::allocateIfNull(DeviceArrayType<V> *arr, sqaod::SizeType size) {
    if (arr->d_data == NULL)
        allocate(arr, size);
}

}

#endif
