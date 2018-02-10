#include "MinimalTestSuite.h"
#include "DeviceTest.h"
#include "DeviceMathTest.h"
#include "CUDAFormulasDGFuncTest.h"
#include "CUDAFormulasBGFuncTest.h"
#include "DeviceRandomTest.h"
#include "CUDADenseGraphBFSolverTest.h"
#include "DeviceSegmentedSumTest.h"
#include "CUDADenseGraphAnnealerTest.h"
#include <iostream>

int main(int argc, char* argv[]) {
    runTest<DeviceTest>();
    runTest<DeviceSegmentedSumTest>();
    runTest<DeviceMathTest>();
    runTest<CUDAFormulasDGFuncTest>();
    runTest<CUDAFormulasBGFuncTest>();
    runTest<DeviceRandomTest>();
    runTest<CUDADenseGraphBFSolverTest>();
    runTest<CUDADenseGraphAnnealerTest>();
    cudaDeviceReset();
    return MinimalTestSuite::summarize();
}
