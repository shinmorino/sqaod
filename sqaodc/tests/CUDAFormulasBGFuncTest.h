#pragma once

#include "MinimalTestSuite.h"
#include <cuda/Device.h>


class CUDAFormulasBGFuncTest : public MinimalTestSuite {
public:
    CUDAFormulasBGFuncTest(void);
    ~CUDAFormulasBGFuncTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();

    sqaod_cuda::Device device_;
};
