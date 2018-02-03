#pragma once

#include "MinimalTestSuite.h"


class CUDADenseGraphAnnealerTest : public MinimalTestSuite {
public:
    CUDADenseGraphAnnealerTest(void);
    ~CUDADenseGraphAnnealerTest(void);

    virtual void setUp();

    virtual void tearDown();

    virtual void run(std::ostream &ostm);

};
