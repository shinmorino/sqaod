#pragma once

#include <sqaodc/common/defines.h>
#include <sqaodc/cuda/Device.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class V>
class DeviceSegmentedSumType {
public:
    DeviceSegmentedSumType(int vecLen);
    virtual ~DeviceSegmentedSumType();

    void configure(sq::SizeType segLen, sq::SizeType nSegments, bool useTempStorage);

protected:
    DeviceSegmentedSumType(Device &device, DeviceStream *devStream, int vecLen);
    DeviceSegmentedSumType(DeviceStream *devStream, int vecLen);

    virtual void chooseKernel() = 0;
};

}
