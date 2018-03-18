#include <sqaodc/sqaodc.h>
#include <iostream>
#include "MinimalTestSuite.h"
#include "BFSearcherRangeCoverageTest.h"

#ifdef SQAODC_CUDA_ENABLED

#include "DeviceTest.h"
#include "DeviceMathTest.h"
#include "CUDAFormulasDGFuncTest.h"
#include "CUDAFormulasBGFuncTest.h"
#include "DeviceRandomTest.h"
#include "CUDADenseGraphBFSolverTest.h"
#include "DeviceSegmentedSumTest.h"
#include "CUDADenseGraphAnnealerTest.h"

#endif

int main(int argc, char* argv[]) {
    
    runTest<BFSearcherRangeCoverageTest>();
#ifdef SQAODC_CUDA_ENABLED
    runTest<DeviceTest>();
    runTest<DeviceSegmentedSumTest>();
    runTest<DeviceMathTest>();
    runTest<CUDAFormulasDGFuncTest>();
    runTest<CUDAFormulasBGFuncTest>();
    runTest<DeviceRandomTest>();
    runTest<CUDADenseGraphBFSolverTest>();
    runTest<CUDADenseGraphAnnealerTest>();
    cudaDeviceReset();
#endif
    return MinimalTestSuite::summarize();
}
