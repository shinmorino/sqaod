#ifndef DEVICE_TEST_H__
#define DEVICE_TEST_H__

#include "MinimalTestSuite.h"
#include <cuda/Device.h>

using namespace sqaod_cuda;

class DeviceTest : public MinimalTestSuite {
public:
    DeviceTest(void);
    ~DeviceTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();
};

#endif
