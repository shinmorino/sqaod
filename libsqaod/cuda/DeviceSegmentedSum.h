#pragma once

#include <common/defines.h>
#include <cuda/Device.h>

namespace sqaod_cuda {

template<class V>
class DeviceSegmentedSumType {
public:
    DeviceSegmentedSumType();
    virtual ~DeviceSegmentedSumType();

    void configure(sqaod::SizeType segLen, sqaod::SizeType nSegments, bool useTempStorage);

protected:
    DeviceSegmentedSumType(Device &device, DeviceStream *devStream);
    DeviceSegmentedSumType(DeviceStream *devStream);

    virtual void chooseKernel() = 0;

    sqaod::SizeType segLen_;
    sqaod::SizeType nSegments_;

    V *d_tempStorage_;
    V *d_tempStoragePreAlloc_;
    sqaod::SizeType tempStorageSize_;
    DeviceStream *devStream_;
    DeviceObjectAllocator *devAlloc_;
};

}
