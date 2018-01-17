#ifndef SQAOD_CUDA_DEVICEOBJECTALLOCATOR_H__
#define SQAOD_CUDA_DEVICEOBJECTALLOCATOR_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceMemoryStore.h>
#include <cuda/DeviceStream.h>

namespace sqaod_cuda {

template<class real>
struct DeviceObjectAllocatorType {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;

    void *allocate(size_t size);

    void deallocate(void *pv);

    void allocate(DeviceMatrix *mat, int rows, int cols);

    void allocate(DeviceMatrix *mat, const sqaod::Dim &dim);

    void allocate(DeviceVector *vec, int size);

    void allocate(DeviceScalar *sc);

    void allocateIfNull(DeviceMatrixType<real> *mat, const sqaod::Dim &dim);

    void allocateIfNull(DeviceVectorType<real> *vec, const sqaod::SizeType size);

    void allocateIfNull(DeviceScalarType<real> *sc);
    
    void deallocate(DeviceObject &obj);

    /* Device Const */
    const DeviceScalar &d_const(const real &c) const;

    const DeviceScalar &d_one() const;

    const DeviceScalar &d_zero() const;

    void initialize(DeviceMemoryStore *memStore, DeviceStream *devStream);
    void finalize();

private:
    DeviceMemoryStore *memStore_;
    static const real hostConsts_[];
    static const int nHostConsts_;
    real *d_consts_;

    typedef sqaod::ArrayType<DeviceScalar*> ConstReg;
    ConstReg constReg_;

    const DeviceScalar *d_one_;
    const DeviceScalar *d_zero_;
};


template<class real>
void DeviceObjectAllocatorType<real>::allocateIfNull(DeviceMatrixType<real> *mat, const sqaod::Dim &dim) {
    if (mat->d_data == NULL)
        allocate(mat, dim);
}

template<class real>
void DeviceObjectAllocatorType<real>::allocateIfNull(DeviceVectorType<real> *vec, const sqaod::SizeType size) {
    if (vec->d_data == NULL)
        allocate(vec, size);
}

template<class real>
void DeviceObjectAllocatorType<real>::allocateIfNull(DeviceScalarType<real> *sc) {
    if (sc->d_data == NULL)
        allocate(sc);
}


template<class real> inline
void DeviceObjectAllocatorType<real>::allocate(DeviceMatrix *mat, const sqaod::Dim &dim) {
    return allocate(mat, dim.rows, dim.cols);
}

template<class real> inline
const DeviceScalarType<real> &DeviceObjectAllocatorType<real>::d_one() const {
    return *d_one_;
}

template<class real> inline
const DeviceScalarType<real> &DeviceObjectAllocatorType<real>::d_zero() const {
    return *d_zero_;
}


}

#endif
