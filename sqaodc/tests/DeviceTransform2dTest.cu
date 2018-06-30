#include "DeviceTransform2dTest.h"
#include <cuda/devfuncs.cuh>
#include <cuda/DeviceCopy.h>
#include <vector>

using namespace sqaod_cuda;

DeviceTransform2dTest::DeviceTransform2dTest(void) : MinimalTestSuite("DeviceTransform2dTest") {
}


DeviceTransform2dTest::~DeviceTransform2dTest(void) {
}


void DeviceTransform2dTest::setUp() {
}

void DeviceTransform2dTest::tearDown() {
}

void DeviceTransform2dTest::run(std::ostream &ostm) {
    Device device;

    device.initialize();
    auto *alloc = device.objectAllocator();
    DeviceStream *defStream = device.defaultStream();
    DeviceObjectAllocator *devAlloc = device.objectAllocator();

    using namespace sqaod_cuda;
    typedef DeviceMatrixType<float> DeviceMatrix;
    typedef sq::MatrixType<float> Matrix;
    DeviceCopy devCopy;

    int nRows = 65536 * 3; /* must be a multiple of 32 */
    int nCols = 8;

    testcase("Transform2dTest") {
        DeviceMatrix dA;
        devAlloc->allocate(&dA, nRows, nCols);
        float *dA_data = dA.d_data;
        sq::SizeType stride = dA.stride;

        auto op = [=]__device__(int gidx, int gidy) {
            dA_data[gidx + gidy * stride] = (float)gidy;
        };
        transform2d(op, nCols, nRows, dim3(8, 16), NULL);

        Matrix A;
        devCopy(&A, dA);
        device.synchronize();

        bool ok = true;
        for (int iRow = 0; iRow < nRows; ++iRow)
            ok &= (A(iRow, 0) == iRow);
        TEST_ASSERT(ok);
    }

    testcase("Transform2dBlockTest") {
        DeviceMatrix dA;

        devAlloc->allocate(&dA, nRows, nCols);
        float *dA_data = dA.d_data;
        sq::SizeType stride = dA.stride;

        auto op = [=]__device__(const dim3 &blockDim, const dim3 &blockIdx, const dim3 &threadIdx) {
            int gidx = blockDim.x * blockIdx.x + threadIdx.x;
            int gidy = blockDim.y * blockIdx.y + threadIdx.y;
            if ((gidx < nCols) && (gidy < nRows))
                dA_data[gidx + gidy * stride] = (float)gidy;
        };
        transformBlock2d(op, divru(nCols, 8), divru(nRows, 32), dim3(8, 32), NULL);

        Matrix A;
        devCopy(&A, dA);
        device.synchronize();

        bool ok = true;
        for (int iRow = 0; iRow < nRows; ++iRow)
            ok &= (A(iRow, 0) == iRow);
        TEST_ASSERT(ok);
    }

    device.finalize();
}
