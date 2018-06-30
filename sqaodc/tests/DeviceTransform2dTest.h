#ifndef DEVICETRANSFORM2D_TEST_H__
#define DEVICETRANSFORM2D_TEST_H__

#include "MinimalTestSuite.h"
#include <cuda/Device.h>


class DeviceTransform2dTest : public MinimalTestSuite {
public:
    DeviceTransform2dTest(void);
    ~DeviceTransform2dTest(void);

    void setUp();

    void tearDown();

    void run(std::ostream &ostm);
private:
    template<class real>
    void tests();
};

#endif
