#ifndef CUDAFORMULASTEST_H__
#define CUDAFORMULASTEST_H__

#include "MinimalTestSuite.h"
#include <cuda/Device.h>

class CUDAFormulasDGFuncTest : public MinimalTestSuite {
public:
    CUDAFormulasDGFuncTest(void);
    ~CUDAFormulasDGFuncTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();

    sqaod_cuda::Device device_;
};

#endif
