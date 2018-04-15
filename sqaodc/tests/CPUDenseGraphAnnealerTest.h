#pragma once

#include "MinimalTestSuite.h"
#include <cpu/CPUDenseGraphAnnealer.h>


class CPUDenseGraphAnnealerTest : public MinimalTestSuite {
public:
    CPUDenseGraphAnnealerTest(void);
    ~CPUDenseGraphAnnealerTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();
};
