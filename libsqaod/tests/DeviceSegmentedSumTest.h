#pragma once

#include "MinimalTestSuite.h"


class DeviceSegmentedSumTest : public MinimalTestSuite {
public:
    DeviceSegmentedSumTest(void);
    ~DeviceSegmentedSumTest(void);

    virtual void setUp();

    virtual void tearDown();

    virtual void run(std::ostream &ostm);

};
