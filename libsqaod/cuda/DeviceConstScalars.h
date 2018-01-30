#pragma once

#include <cuda/DeviceObjectAllocator.h>
#include <cuda/DeviceStream.h>
#include <cuda/DeviceMatrix.h>

namespace sqaod_cuda {


template<class real>
struct DeviceConstScalarsType {
    DeviceConstScalarsType();
    ~DeviceConstScalarsType();

    typedef DeviceScalarType<real> DeviceScalar;

    
    /* Device Const */
    const DeviceScalar &d_const(const real &c) const;

    const DeviceScalar &d_one() const;

    const DeviceScalar &d_zero() const;

    void initialize(DeviceObjectAllocator &devAlloc, DeviceStream &devStream);
    void finalize(DeviceObjectAllocator &devAlloc);

private:    
    static const real hostConsts_[];
    static const int nHostConsts_;
    real *d_consts_;

    typedef sqaod::ArrayType<DeviceScalar*> ConstReg;
    ConstReg constReg_;

    const DeviceScalar *d_one_;
    const DeviceScalar *d_zero_;

    /* prohibit copy c-tor. */
    DeviceConstScalarsType(const DeviceConstScalarsType<real> &);
};


template<class real> inline
const DeviceScalarType<real> &DeviceConstScalarsType<real>::d_one() const {
    return *d_one_;
}

template<class real> inline
const DeviceScalarType<real> &DeviceConstScalarsType<real>::d_zero() const {
    return *d_zero_;
}

}
