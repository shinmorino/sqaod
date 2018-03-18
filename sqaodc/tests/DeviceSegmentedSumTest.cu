#include "DeviceSegmentedSumTest.h"
#include <cuda_runtime.h>
#include <cuda/DeviceSegmentedSum.cuh>
#include <cuda/DeviceCopy.h>
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

template<class V>
void DeviceSegmentedSumTest::runSegmentedSum(int segLen, int nSegments) {
    sqcu::DeviceObjectAllocator *alloc = device_.objectAllocator();
    sqcu::DeviceCopy copy(device_);

    typedef sqcu::DeviceMatrixType<V> DeviceMatrix;
    typedef sqcu::DeviceVectorType<V> DeviceVector;
    typedef sq::MatrixType<V> HostMatrix;
    typedef sq::VectorType<V> HostVector;

    typedef sqcu::DeviceSegmentedSumTypeImpl<V, V*, V*, sqcu::Linear> SegmentedSum;

    testcase("SegmentedSum") {
        SegmentedSum segSum(device_);
        segSum.configure(segLen, nSegments, false);
        sq::Dim dim(nSegments, segLen);

        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMatBalanced<V>(dim);
        HostVector x = segmentedSum(A, segLen, nSegments);

        copy(&dA, A);
        alloc->allocate(&dx, nSegments);
        segSum.configure(segLen, nSegments, false);
        segSum(dA.d_data, dx.d_data, sqcu::Linear(segLen, 0));
        device_.synchronize();

        TEST_ASSERT(allclose(dx, x, epusiron<V>()));

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }
}

template<class V>
void DeviceSegmentedSumTest::test() {
#if 1
    typedef sqcu::DeviceSegmentedSumTypeImpl<V, V*, V*, sqcu::Linear> SegmentedSum;
    SegmentedSum segSum(device_);
    for (typename SegmentedSum::MethodMap::iterator it = segSum.methodMap_.begin();
        it != segSum.methodMap_.end(); ++it) {
        runSegmentedSum<V>(it->first, it->first / 8);
    }
#else
    runSegmentedSum<V>(12288, 2);
#endif
}

void DeviceSegmentedSumTest::run(std::ostream &ostm) {
    test<float>();
    test<double>();
}
