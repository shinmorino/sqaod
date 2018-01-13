#include "DeviceObjectAllocator.h"

using namespace sqaod_cuda;
using sqaod::Dim;

template<class real>
const real DeviceObjectAllocatorType<real>::hostConsts_[] = {0., 0.25, 0.5, 1.};
template<class real>
const int DeviceObjectAllocatorType<real>::nHostConsts_ = sizeof(hostConsts_) / sizeof(real);


template<class real>
void *DeviceObjectAllocatorType<real>::allocate(size_t size) {
    return memStore_->allocate(size);
}

template<class real>
void DeviceObjectAllocatorType<real>::deallocate(void *pv) {
    return memStore_->deallocate(pv);
}

template<class real>
void DeviceObjectAllocatorType<real>::allocate(DeviceMatrix *mat, int rows, int cols) {
    mat->d_data = (real*)memStore_->allocate(sizeof(real) * rows * cols);
    mat->rows = rows;
    mat->cols = cols;
}

template<class real>
void DeviceObjectAllocatorType<real>::allocate(DeviceVectorType<real> *vec, int size) {
    vec->d_data = (real*)memStore_->allocate(sizeof(real) * size);
    vec->size = size;
}

template<class real>
void DeviceObjectAllocatorType<real>::allocate(DeviceScalarType<real> *sc) {
    sc->d_data = (real*)memStore_->allocate(sizeof(real));
}

template<class real>
void DeviceObjectAllocatorType<real>::deallocate(DeviceObject &obj) {
    void *pv = obj.get_data();
    memStore_->deallocate(pv);
}

    
/* Device Const */
template<class real>
const DeviceScalarType<real> &DeviceObjectAllocatorType<real>::d_const(const real &c) const {
    if (c == 1.)
        return *d_one_;
    if (c == 0.)
        return *d_zero_;
    const real *pos = std::find(hostConsts_, hostConsts_ + nHostConsts_, c);
    int idx = int(pos - hostConsts_);
    THROW_IF(idx == nHostConsts_, "Constant not registered.");
    return *constReg_[idx];
}

template<class real>
void DeviceObjectAllocatorType<real>::initialize(DeviceMemoryStore *memStore,
                                                 DeviceStream *devStream) {
    memStore_ = memStore;
    
    d_consts_ = (real*)memStore_->allocate(sizeof(real) * nHostConsts_);
    DeviceCopy(devStream).copy(d_consts_, hostConsts_, nHostConsts_);

    for (int idx = 0; idx < nHostConsts_; ++idx)
        constReg_.pushBack(new DeviceScalar(&d_consts_[idx]));
    d_one_ = &d_const(1.);
    d_zero_ = &d_const(0.);
}

template<class real>
void DeviceObjectAllocatorType<real>::finalize() {
    for (int idx = 0; idx < nHostConsts_; ++idx)
        delete constReg_[idx];
    
    constReg_.clear();
    memStore_->deallocate(d_consts_);
    d_consts_ = NULL;
}

template struct sqaod_cuda::DeviceObjectAllocatorType<float>;
template struct sqaod_cuda::DeviceObjectAllocatorType<double>;
