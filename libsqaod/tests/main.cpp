#include "MinimalTestSuite.h"
#include "DeviceTest.h"
#include "DeviceMathTest.h"
#include <iostream>

int main(int argc, char* argv[]) {
    runTest<DeviceTest>();
    runTest<DeviceMathTest>();
    return MinimalTestSuite::summarize();
}
