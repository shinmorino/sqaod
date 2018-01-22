#ifndef TESTS_H__
#define TESTS_H__

#include "MinimalTestSuite.h"
#include <cuda/Device.h>

class DeviceMathTest : public MinimalTestSuite {
public:
    DeviceMathTest(void);
    ~DeviceMathTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests(const sqaod::Dim &dim);

    sqaod_cuda::Device device_;
};

#endif
