#include "MinimalTestSuite.h"
#include <iostream>

int MinimalTestSuite::okCount_ = 0;
int MinimalTestSuite::failCount_ = 0;

void MinimalTestSuite::success() {
    std::cerr << "." << std::flush;
    ++okCount_;
}

void MinimalTestSuite::fail(const char *filename, unsigned long lineno) {
    std::cerr << filename << ":" << lineno << ": Test failed(" << testName_ << ")" << std::endl;
    ++failCount_;
}

int MinimalTestSuite::summarize() {
    std::cerr << std::endl
              << std::endl
              << "FAILED: " << failCount_ << " / ALL: " << okCount_ + failCount_ << std::endl
              << std::endl;

    if (failCount_ == 0)
        std::cerr << "PASSED ALL TESTS." << std::endl;
    std::cerr << std::endl;

    if (failCount_ != 0)
        return 1;
    return 0;
}

