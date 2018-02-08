#pragma once

#include "MinimalTestSuite.h"
#include <cuda/Device.h>

class CUDADenseGraphAnnealerTest : public MinimalTestSuite {
public:
    CUDADenseGraphAnnealerTest(void);
    ~CUDADenseGraphAnnealerTest(void);

    virtual void setUp();

    virtual void tearDown();

    virtual void run(std::ostream &ostm);

private:
    template<class real>
    void test();

    sqaod_cuda::Device device_;

};
