#include "DeviceConstScalars.h"
#include "DeviceCopy.h"

using namespace sqaod_cuda;


template<class real>
const real DeviceConstScalarsType<real>::hostConsts_[] = {0., 0.25, 0.5, 1.};
template<class real>
const int DeviceConstScalarsType<real>::nHostConsts_ = sizeof(hostConsts_) / sizeof(real);

template<class real>
DeviceConstScalarsType<real>::DeviceConstScalarsType() : d_consts_(NULL), d_one_(NULL), d_zero_(NULL) {
}

template<class real>
DeviceConstScalarsType<real>::~DeviceConstScalarsType() {
}

    
/* Device Const */
template<class real>
const DeviceScalarType<real> &DeviceConstScalarsType<real>::d_const(const real &c) const {
    if (c == 1.)
        return *d_one_;
    if (c == 0.)
        return *d_zero_;
    const real *pos = std::find(hostConsts_, hostConsts_ + nHostConsts_, c);
    int idx = int(pos - hostConsts_);
    abortIf(idx == nHostConsts_, "Constant not registered.");
    return *constReg_[idx];
}

template<class real>
void DeviceConstScalarsType<real>::initialize(DeviceObjectAllocator &devAlloc,
                                              DeviceStream &devStream) {
    d_consts_ = (real*)devAlloc.allocate(sizeof(real) * nHostConsts_);
    DeviceCopy().copy(d_consts_, hostConsts_, nHostConsts_);

    for (int idx = 0; idx < nHostConsts_; ++idx)
        constReg_.pushBack(new DeviceScalar(&d_consts_[idx]));

    const real *pos = std::find(hostConsts_, hostConsts_ + nHostConsts_, 0.);
    int idx = int(pos - hostConsts_);
    d_zero_ = constReg_[idx];
    pos = std::find(hostConsts_, hostConsts_ + nHostConsts_, 1.);
    idx = int(pos - hostConsts_);
    d_one_ = constReg_[idx];
}

template<class real>
void DeviceConstScalarsType<real>::finalize(DeviceObjectAllocator &devAlloc) {
    for (int idx = 0; idx < nHostConsts_; ++idx)
        delete constReg_[idx];
    
    constReg_.clear();
    devAlloc.deallocate(d_consts_);
    d_consts_ = NULL;
}

template struct sqaod_cuda::DeviceConstScalarsType<float>;
template struct sqaod_cuda::DeviceConstScalarsType<double>;
