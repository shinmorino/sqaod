#include "DeviceSegmentedSum.h"

using namespace sqaod_cuda;
namespace sq = sqaod;

template<class V>
DeviceSegmentedSumType<V>::DeviceSegmentedSumType() {
    d_tempStoragePreAlloc_ = NULL;
    d_tempStorage_ = NULL;
    devStream_ = NULL;
    devAlloc_ = NULL;
    segLen_ = 0;
    nSegments_ = 0;
}

template<class V>
DeviceSegmentedSumType<V>::~DeviceSegmentedSumType() {
    if (d_tempStoragePreAlloc_ != NULL)
        devAlloc_->deallocate(d_tempStoragePreAlloc_);
}

template<class V>
DeviceSegmentedSumType<V>::DeviceSegmentedSumType(Device &device, DeviceStream *devStream) {
    d_tempStoragePreAlloc_ = NULL;
    segLen_ = 0;
    nSegments_ = 0;

    if (devStream == NULL)
        devStream = device.defaultStream();
    devStream_ = devStream;
    devAlloc_ = device.objectAllocator();
}

template<class V>
DeviceSegmentedSumType<V>::DeviceSegmentedSumType(DeviceStream *devStream) {
    d_tempStoragePreAlloc_ = NULL;
    devAlloc_ = NULL;
    segLen_ = 0;
    nSegments_ = 0;

    devStream_ = devStream;
}

template<class V>
void DeviceSegmentedSumType<V>::configure(sq::SizeType segLen, sq::SizeType nSegments, bool useTempStorage) {
    segLen_ = segLen;
    nSegments_ = nSegments;
    chooseKernel();
    d_tempStorage_ = NULL;
    tempStorageSize_ = 0;
    if (4096 < segLen) {
        tempStorageSize_ = 32;
        if (!useTempStorage)
            devAlloc_->allocate(&d_tempStoragePreAlloc_, tempStorageSize_);
    }
}

template class DeviceSegmentedSumType<double>;
template class DeviceSegmentedSumType<float>;

