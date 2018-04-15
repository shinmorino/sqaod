#pragma once

#include "MinimalTestSuite.h"
#include <cpu/CPUBipartiteGraphAnnealer.h>


class CPUBipartiteGraphAnnealerTest : public MinimalTestSuite {
public:
    CPUBipartiteGraphAnnealerTest(void);
    ~CPUBipartiteGraphAnnealerTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();
};
