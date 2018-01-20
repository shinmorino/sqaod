#ifndef MINIMAL_TESTSUITE_H__
#define MINIMAL_TESTSUITE_H__

#include <string>
#include <iostream>


class MinimalTestSuite {
public:
    MinimalTestSuite(const std::string &testName)
        : testName_(testName) {  }

    void reset(int curNo) {
        testNo_ = 0;
        curNo_ = curNo;
    }
    
    void success();

    void fail(const char *filename, unsigned long lineno);

protected:
    int testNo_;
    int curNo_;

private:
    std::string testName_;

/* static members */
public:
    static
    int summarize();

    template<class T>
    friend void runTest();
    
private:
    static int okCount_;
    static int failCount_;
};

template<class T>
void runTest() {
    T t;

    t.reset(-1);
    t.run(std::cerr);
    int nTests = t.testNo_;
    
    for (t.curNo_ = 0; t.curNo_ < nTests; ++t.curNo_) {
        t.setUp();
        t.reset(t.curNo_);
        t.run(std::cerr);
        t.tearDown();
    }
}


#define TEST_ASSERT(x) { if (x) success(); else fail(__FILE__, __LINE__); }
#define TEST_SUCCESS   { success(); }
#define TEST_FAIL      { fail(__FILE__, __LINE__); }
#define testcase(name) if ((testNo_++) == curNo_)


#endif
