#ifndef INTEGRALIMAGE_TESTS_H__
#define INTEGRALIMAGE_TESTS_H__

#include "LibImageProcDemoTests/MinimalTestSuite.h"
#include <LibImageProcDemo/Image.h>
#include <LibImageProcDemo/IntegralImage.h>


class IntegralImageTests : public MinimalTestSuite {
public:
    IntegralImageTests(void);
    ~IntegralImageTests(void);

    virtual void setUp();

    virtual void tearDown();
    
    virtual void run(std::ostream &ostm);

private:
    HostImage<float> image_;
    IntegralImage integralImage_;
};

#endif
