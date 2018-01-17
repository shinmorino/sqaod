#include "Test.h"
#include <iostream>

int main(int argc, char* argv[]) {
    runTest<Test>();

    return MinimalTestSuite::summarize();
}
