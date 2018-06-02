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
    template<class V, class SegmentedSum>
    void runSegmentedSum(int segLen, int nSegments, bool clearPadding);
    template<class V>
    void test();

    sqaod_cuda::Device device_;
};
