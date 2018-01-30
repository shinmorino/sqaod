#pragma once

#include "MinimalTestSuite.h"
#include <cuda/Device.h>


class DeviceRandomTest : public MinimalTestSuite {
public:
    DeviceRandomTest(void);
    ~DeviceRandomTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);

private:
    sqaod_cuda::Device device_;
};
