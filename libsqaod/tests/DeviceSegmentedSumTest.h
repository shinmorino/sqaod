#pragma once

#include "MinimalTestSuite.h"
#include <cuda/Device.h>

class DeviceSegmentedSumTest : public MinimalTestSuite {
public:
    DeviceSegmentedSumTest(void);
    ~DeviceSegmentedSumTest(void);

    virtual void setUp();

    virtual void tearDown();

    virtual void run(std::ostream &ostm);

private:
    sqaod_cuda::Device device_;
};
