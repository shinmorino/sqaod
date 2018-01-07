#ifndef SQAOD_CUDA_DEVICEOBJECTALLOCATOR_H__
#define SQAOD_CUDA_DEVICEOBJECTALLOCATOR_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceCopy.h>
#include <cuda/DeviceMemoryStore.h>

namespace sqaod_cuda {

template<class real>
struct DeviceObjectAllocatorType {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    
    void allocate(DeviceMatrix *mat, int rows, int cols);

    void allocate(DeviceMatrix *mat, const sqaod::Dim &dim);

    void allocate(DeviceVector *vec, int size);

    void allocate(DeviceScalar *mat);
    
    void deallocate(DeviceObject &obj);
    
    /* Device Const */
    const DeviceScalar &d_const(const real &c) const;

    const DeviceScalar &d_one() const;

    const DeviceScalar &d_zero() const;

    void initialize(DeviceMemoryStore &memStore, DeviceStream &stream);
    void finalize();

private:
    DeviceMemoryStore *memStore_;
    DeviceCopy devCopy_;
    static const real hostConsts_[];
    static const int nHostConsts_;
    real *d_consts_;

    typedef sqaod::ArrayType<DeviceScalar*> ConstReg;
    ConstReg constReg_;

    const DeviceScalar *d_one_;
    const DeviceScalar *d_zero_;
};


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
