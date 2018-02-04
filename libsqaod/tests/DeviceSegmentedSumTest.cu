#include "DeviceSegmentedSumTest.h"
#include <cuda_runtime.h>
#include <cuda/DeviceSegmentedSum.cuh>
#include <cuda/DeviceCopy.h>
//#include <common/Matrix.h>
#include "utils.h"

using namespace sqaod_cuda;
namespace sq = sqaod;

DeviceSegmentedSumTest::DeviceSegmentedSumTest(void) : MinimalTestSuite("DeviceSegmentedSumTest")
{
}


DeviceSegmentedSumTest::~DeviceSegmentedSumTest(void)
{
}


void DeviceSegmentedSumTest::setUp() {
    device_.useManagedMemory(true);
    device_.initialize();
}

void DeviceSegmentedSumTest::tearDown() {
    device_.finalize();
}

void DeviceSegmentedSumTest::run(std::ostream &ostm) {
    DeviceObjectAllocator *alloc = device_.objectAllocator();
    DeviceCopy copy(device_);

    typedef float real;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;

    testcase("SegmentedSum_32") {
        sq::Dim dim(32, 32);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_32(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_32 2") {
        sq::Dim dim(185, 32);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_32(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_32 3") {
        sq::Dim dim(185, 18);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_32(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_64") {
        sq::Dim dim(64, 64);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_64(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_64 2") {
        sq::Dim dim(185, 64);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_64(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_64 3") {
        sq::Dim dim(185, 40);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_64(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128") {
        sq::Dim dim(128, 128);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_128(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128 2") {
        sq::Dim dim(300, 128);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_128(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128 3") {
        sq::Dim dim(300, 100);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_128(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }


    testcase("SegmentedSum_128Loop") {
        sq::Dim dim(32, 1024);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_128Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 1, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128 2") {
        sq::Dim dim(300, 1024);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_128Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 1, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128Loop 3") {
        sq::Dim dim(32, 500);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_128Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 1, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128Loop 2 blocks / seq") {
        sq::Dim dim(32, 1024);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = HostMatrix::ones(dim);
        HostVector x(dim.rows * 2);
        x = real(512.);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows * 2);
        segmentedSum_128Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 2, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128Loop 2 blocks / seq 2") {
        sq::Dim dim(32, 3000);
        DeviceMatrix dA;
        DeviceVector dx;

        // HostMatrix A = HostMatrix::ones(dim);
        HostMatrix A = testMat<real>(dim);
        HostVector x = HostVector::zeros(dim.rows * 2);
        for (int iSeg = 0; iSeg < (int)dim.rows; ++iSeg) {
            for (int segPos = 0; segPos < (int)dim.cols; segPos += 128 * 2) {
                for (int pos = segPos; pos < std::min((int)dim.cols, segPos + 128); ++pos)
                    x(iSeg * 2) += A(iSeg, pos);
                for (int pos = segPos + 128; pos < std::min((int)dim.cols, segPos + 256); ++pos)
                    x(iSeg * 2 + 1) += A(iSeg, pos);
            }
        }

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows * 2);
        segmentedSum_128Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 2, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_128Loop 3 blocks / seq") {
        sq::Dim dim(32, 3000);
        DeviceMatrix dA;
        DeviceVector dx;

        // HostMatrix A = HostMatrix::ones(dim);
        HostMatrix A = testMat<real>(dim);
        HostVector x = HostVector::zeros(dim.rows * 3);
        for (int iSeg = 0; iSeg < (int)dim.rows; ++iSeg) {
            for (int segPos = 0; segPos < (int)dim.cols; segPos += 128 * 3) {
                for (int pos = segPos; pos < std::min((int)dim.cols, segPos + 128); ++pos)
                    x(iSeg * 3) += A(iSeg, pos);
                for (int pos = segPos + 128; pos < std::min((int)dim.cols, segPos + 256); ++pos)
                    x(iSeg * 3 + 1) += A(iSeg, pos);
                for (int pos = segPos + 256; pos < std::min((int)dim.cols, segPos + 384); ++pos)
                    x(iSeg * 3 + 2) += A(iSeg, pos);
            }
        }

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows * 3);
        segmentedSum_128Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 3, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_512Loop") {
        sq::Dim dim(32, 2048);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_512Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum_512Loop 2") {
        sq::Dim dim(32, 768);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum_512Loop(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum, big") {
        sq::Dim dim(2, 4096);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);

        sq::SizeType temp_storage_bytes;
        segmentedSum(NULL, &temp_storage_bytes, dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 50000, NULL);
        if (temp_storage_bytes != 0) {
            void *temp_storage = alloc->allocate(temp_storage_bytes);
            segmentedSum(temp_storage, &temp_storage_bytes, dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, 50000, NULL);
        }
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }

    testcase("SegmentedSum, no temp mem") {
        sq::Dim dim(30, 4096);
        DeviceMatrix dA;
        DeviceVector dx;

        HostMatrix A = testMat<real>(dim);
        HostVector x = segmentedSum(A, dim.cols, dim.rows);

        copy(&dA, A);
        alloc->allocate(&dx, dim.rows);
        segmentedSum(dA.d_data, dx.d_data, Linear(dim.cols, 0), dim.cols, dim.rows, NULL);
        device_.synchronize();

        TEST_ASSERT(dx == x);

        alloc->deallocate(dA);
        alloc->deallocate(dx);
    }
}
