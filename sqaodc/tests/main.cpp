#include <sqaodc/sqaodc.h>
#include <iostream>
#include "MinimalTestSuite.h"
#include "BFSearcherRangeCoverageTest.h"
#include "CPUDenseGraphAnnealerTest.h"
#include "CPUBipartiteGraphAnnealerTest.h"

#ifdef SQAODC_CUDA_ENABLED

#include "DeviceTest.h"
#include "DeviceTransform2dTest.h"
#include "DeviceMathTest.h"
#include "CUDAFormulasDGFuncTest.h"
#include "CUDAFormulasBGFuncTest.h"
#include "DeviceRandomTest.h"
#include "CUDADenseGraphBFSolverTest.h"
#include "DeviceSegmentedSumTest.h"
#include "CUDADenseGraphAnnealerTest.h"
#include "CUDABipartiteGraphAnnealerTest.h"

#endif

int main(int argc, char* argv[]) {
    
    runTest<BFSearcherRangeCoverageTest>();
    runTest<CPUDenseGraphAnnealerTest>();
    runTest<CPUBipartiteGraphAnnealerTest>();
#ifdef SQAODC_CUDA_ENABLED
    runTest<DeviceTest>();
    runTest<DeviceTransform2dTest>();
    runTest<DeviceSegmentedSumTest>();
    runTest<DeviceMathTest>();
    runTest<CUDAFormulasDGFuncTest>();
    runTest<CUDAFormulasBGFuncTest>();
    runTest<DeviceRandomTest>();
    runTest<CUDADenseGraphBFSolverTest>();
    runTest<CUDADenseGraphAnnealerTest>();
    runTest<CUDABipartiteGraphAnnealerTest>();
    cudaDeviceReset();
#endif
    return MinimalTestSuite::summarize();
}
