#include "CUDAFormulasDGFuncTest.h"
#include <cuda/CUDAFormulas.h>
#include <cpu/CPUFormulas.h>
#include <stdlib.h>
#include "utils.h"

namespace sq = sqaod;


CUDAFormulasDGFuncTest::CUDAFormulasDGFuncTest(void) : MinimalTestSuite("CUDAFormulasDGFuncTest") {
}


CUDAFormulasDGFuncTest::~CUDAFormulasDGFuncTest(void) {
}


void CUDAFormulasDGFuncTest::setUp() {
    device_.initialize();
}

void CUDAFormulasDGFuncTest::tearDown() {
    device_.finalize();
}
    
void CUDAFormulasDGFuncTest::run(std::ostream &ostm) {
    tests<double>();
    tests<float>();
}

template<class real>
void CUDAFormulasDGFuncTest::tests() {

    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    // typedef sq::EigenMatrixType<real> EigenMatrix;
    // typedef sq::EigenRowVectorType<real> EigenRowVector;
    // typedef sq::EigenColumnVectorType<real> EigenColumnVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;

    DeviceCopy devCopy(device_);
    // DeviceStream *devStream = device_.defaultStream();
    // Device::ObjectAllocator *alloc = device_.objectAllocator();

    typedef sq::DGFuncs<real> DGF;
    CUDADGFuncs<real> devFuncs;
    devFuncs.assignDevice(device_);

    const int N = 120;
    const int m = 100;

    testcase("calculate_E") {
        HostMatrix W = testMatSymmetric<real>(N);
        HostVector x = randomizeBits<real>(N);
        real E;
        DGF::calculate_E(&E, W, x);

        DeviceMatrix dW;
        DeviceVector dx;
        DeviceScalar dE;
        devCopy(&dW, W);
        devCopy(&dx, x);
        devFuncs.calculate_E(&dE, dW, dx);
        device_.synchronize();
        TEST_ASSERT(dE == E);
    }

    testcase("calculate_E batched") {
        HostMatrix W = testMatSymmetric<real>(N);
        HostMatrix X = randomizeBits<real>(sq::Dim(m, N));
        HostVector E(m);
        DGF::calculate_E(&E, W, X);

        DeviceMatrix dW;
        DeviceMatrix dX;
        DeviceVector dE;
        devCopy(&dW, W);
        devCopy(&dX, X);
        devFuncs.calculate_E(&dE, dW, dX);
        device_.synchronize();
        TEST_ASSERT(dE == E);
    }

    testcase("calulcate_hJc") {
        HostVector h;
        HostMatrix J;
        real c;
        HostMatrix W = testMatSymmetric<real>(N);
        DGF::calculate_hJc(&h, &J, &c, W);

        DeviceMatrix dW;
        devCopy(&dW, W);
        
        DeviceMatrix dJ;
        DeviceVector dh;
        DeviceScalar dc;
        devFuncs.calculate_hJc(&dh, &dJ, &dc, dW);

        TEST_ASSERT(dh == h);
        TEST_ASSERT(dJ == J);
        TEST_ASSERT(dc == c);


    }

    testcase("calulcate_hJc") {
        HostMatrix W = testMatSymmetric<real>(N);
        HostVector h, q;
        HostMatrix J;
        real c;
        q = randomizeBits<real>(N);
        DGF::calculate_hJc(&h, &J, &c, W);
        real E;
        DGF::calculate_E(&E, h, J, c, q);

        DeviceMatrix dW;
        DeviceMatrix dJ;
        DeviceVector dh, dq;
        DeviceScalar dc;
        devCopy(&dW, W);
        devCopy(&dq, q);
        devFuncs.calculate_hJc(&dh, &dJ, &dc, dW);

        DeviceScalar dE;
        devFuncs.calculate_E(&dE, dh, dJ, dc, dq);
        TEST_ASSERT(dE == E);
    }

    testcase("calulcate_hJc") {
        HostMatrix W = testMatSymmetric<real>(N);
        HostVector h;
        HostMatrix J;
        real c;
        HostMatrix q = randomizeBits<real>(sqaod::Dim(m, N));
        DGF::calculate_hJc(&h, &J, &c, W);
        HostVector E;
        DGF::calculate_E(&E, h, J, c, q);

        DeviceMatrix dW;
        DeviceMatrix dJ, dq;
        DeviceVector dh;
        DeviceScalar dc;
        devCopy(&dW, W);
        devCopy(&dq, q);
        devFuncs.calculate_hJc(&dh, &dJ, &dc, dW);

        DeviceVector dE;
        devFuncs.calculate_E(&dE, dh, dJ, dc, dq);
        TEST_ASSERT(dE == E);
    }

}
