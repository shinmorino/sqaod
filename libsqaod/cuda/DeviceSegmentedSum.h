#pragma once

#include <common/defines.h>
#include <cuda/Device.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class V>
class DeviceSegmentedSumType {
public:
    DeviceSegmentedSumType();
    virtual ~DeviceSegmentedSumType();

    void configure(sq::SizeType segLen, sq::SizeType nSegments, bool useTempStorage);

protected:
    DeviceSegmentedSumType(Device &device, DeviceStream *devStream);
    DeviceSegmentedSumType(DeviceStream *devStream);

    virtual void chooseKernel() = 0;

    sq::SizeType segLen_;
    sq::SizeType nSegments_;

    V *d_tempStorage_;
    V *d_tempStoragePreAlloc_;
    sq::SizeType tempStorageSize_;
    DeviceStream *devStream_;
    DeviceObjectAllocator *devAlloc_;
};

}
