#ifndef TESTS_H__
#define TESTS_H__

#include "MinimalTestSuite.h"
#include <cuda/Device.h>

using namespace sqaod_cuda;

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

    Device device_;
};

#endif
