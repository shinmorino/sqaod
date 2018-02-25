#pragma once

#include "MinimalTestSuite.h"
#include <cuda/CUDADenseGraphBFSearcher.h>
#include <cuda/Device.h>

class CUDADenseGraphBFSolverTest : public MinimalTestSuite {
public:
    CUDADenseGraphBFSolverTest(void);
    ~CUDADenseGraphBFSolverTest(void);

    virtual void setUp();

    virtual void tearDown();
    
    virtual void run(std::ostream &ostm);
private:
    template<class real>
    void tests();

    sqaod_cuda::Device device_;
};
