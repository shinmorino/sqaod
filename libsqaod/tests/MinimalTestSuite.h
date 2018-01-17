#ifndef MINIMAL_TESTSUITE_H__
#define MINIMAL_TESTSUITE_H__

#include <string>
#include <iosfwd>
#include <LibImageProcDemo/Image.h>

class MinimalTestSuite {
public:
    MinimalTestSuite(const std::string &testName)
        : testName_(testName) { }

    virtual void setUp() { }

    virtual void tearDown() { }
    
    virtual void run(std::ostream &ostm) { }

    void success();

    void fail(const char *filename, unsigned long lineno);

private:
    std::string testName_;


/* static members */
public:
    static
    int summarize();

private:
    static int okCount_;
    static int failCount_;
};

template<class T>
void runTest() {
    T t;
    t.setUp();
    t.run(std::cerr);
    t.tearDown();
}



bool compare(const DeviceImage<float> &gpu, const HostImage<float> &cpu, std::ostream &ostm);
bool compare(const char *caption, const DeviceImage<float> &gpu, const HostImage<float> &cpu, float tolerance, std::ostream &ostm);


inline
float normErr(float vObs, float vExp) {
    return (fabs(vObs / vExp - 1.f));
}


#define TEST_ASSERT(x) { if (x) success(); else fail(__FILE__, __LINE__); }


#endif
