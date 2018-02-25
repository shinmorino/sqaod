#include "DeviceObjectAllocator.h"
#include "DeviceCopy.h"

using namespace sqaod_cuda;


void *DeviceObjectAllocator::allocate(size_t size) {
    return memStore_->allocate(size);
}

void DeviceObjectAllocator::deallocate(void *pv) {
    return memStore_->deallocate(pv);
}

void DeviceObjectAllocator::deallocate(DeviceObject &obj) {
    void *pv;
    obj.get_data(&pv);
    memStore_->deallocate(pv);
}

