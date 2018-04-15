#pragma once

#include "MinimalTestSuite.h"
#include <cuda/CUDABipartiteGraphAnnealer.h>


class CUDABipartiteGraphAnnealerTest : public MinimalTestSuite {
public:
    CUDABipartiteGraphAnnealerTest(void);
    ~CUDABipartiteGraphAnnealerTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();

    sqaod_cuda::Device device_;
};
