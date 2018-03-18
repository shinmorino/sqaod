#pragma once

#include "MinimalTestSuite.h"
#include <sqaodc/sqaodc.h>


class BFSearcherRangeCoverageTest : public MinimalTestSuite {
public:
    BFSearcherRangeCoverageTest(void);
    ~BFSearcherRangeCoverageTest(void);

    void setUp();

    void tearDown();
    
    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();
};
