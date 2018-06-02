#include "DeviceSegmentedSumTest.h"
#include <cuda_runtime.h>
#include <cuda/DeviceSegmentedSum.cuh>
#include <cuda/DeviceCopy.h>
#include <cuda/DeviceBatchedDot.cuh>
//#include <common/Matrix.h>
#include "utils.h"

namespace sqcu = sqaod_cuda;

DeviceSegmentedSumTest::DeviceSegmentedSumTest(void) : MinimalTestSuite("DeviceSegmentedSumTest")
{
}


DeviceSegmentedSumTest::~DeviceSegmentedSumTest(void)
{
}


void DeviceSegmentedSumTest::setUp() {
    device_.useManagedMemory(true);
    device_.enableLocalStore(false);
    device_.initialize();
}

void DeviceSegmentedSumTest::tearDown() {
    device_.finalize();
}

template<class V, class SegmentedSum>
void DeviceSegmentedSumTest::runSegmentedSum(int segLen, int nSegments, bool clearPadding) {
    sqcu::DeviceObjectAllocator *alloc = device_.objectAllocator();
    sqcu::DeviceCopy copy(device_);

    typedef sqcu::DeviceMatrixType<V> DeviceMatrix;
    typedef sqcu::DeviceVectorType<V> DeviceVector;
    typedef sq::MatrixType<V> HostMatrix;
    typedef sq::VectorType<V> HostVector;

    testcase("SegmentedSum") {
        SegmentedSum segSum(device_);
        segSum.configure(segLen, nSegments, false);
        sq::Dim dim(nSegments, segLen);

        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMatBalanced<V>(dim);
        HostVector x = segmentedSum(A, segLen, nSegments);

        copy(&dA, A);
        if (clearPadding)
            copy.clearPadding(&dA);
        alloc->allocate(&dx, nSegments);
        segSum.configure(segLen, nSegments, false);
        segSum(dA, dx.d_data);
        device_.synchronize();

        TEST_ASSERT(allclose(dx, x, epusiron<V>()));

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }
}

template<class V>
void DeviceSegmentedSumTest::test() {
    typedef sqcu::DeviceBatchedSum<V, V*> DeviceBatchedSum;
    typedef sqcu::DeviceBatchedSumVec4<V, V*> DeviceBatchedSumVec4;
#if 1
    DeviceBatchedSum segSum(device_);
    for (typename DeviceBatchedSum::MethodMap::iterator it = segSum.methodMap_.begin();
        it != segSum.methodMap_.end(); ++it) {
        runSegmentedSum<V, DeviceBatchedSum>(it->first, it->first / 8, false);
    }

    DeviceBatchedSumVec4 segSumVec4(device_);
    for (typename DeviceBatchedSumVec4::MethodMap::iterator it4 = segSumVec4.methodMap_.begin();
        it4 != segSumVec4.methodMap_.end(); ++it4) {
        int segLen = it4->first * 4;
        runSegmentedSum<V, DeviceBatchedSumVec4>(segLen, segLen / 8, false);
    }

    for (typename DeviceBatchedSumVec4::MethodMap::iterator it4 = segSumVec4.methodMap_.begin();
        it4 != segSumVec4.methodMap_.end(); ++it4) {
        int segLen = it4->first * 4;
        runSegmentedSum<V, DeviceBatchedSumVec4>(segLen - 2, segLen / 8, true);
    }
#else
    runSegmentedSum<V, DeviceBatchedSumVec4>(128, 2);
#endif
}

void DeviceSegmentedSumTest::run(std::ostream &ostm) {
    test<float>();
    test<double>();
}
