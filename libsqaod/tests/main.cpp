#include "MinimalTestSuite.h"
#include "DeviceTest.h"
#include "DeviceMathTest.h"
#include "CUDAFormulasDGFuncTest.h"
#include "CUDAFormulasBGFuncTest.h"
#include <iostream>

int main(int argc, char* argv[]) {
    runTest<DeviceTest>();
    runTest<DeviceMathTest>();
    runTest<CUDAFormulasDGFuncTest>();
    runTest<CUDAFormulasBGFuncTest>();
    return MinimalTestSuite::summarize();
}
