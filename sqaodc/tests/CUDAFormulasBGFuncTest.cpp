#include "CUDAFormulasBGFuncTest.h"
#include <cuda/DeviceFormulas.h>
#include <cpu/CPUFormulas.h>
#include <stdlib.h>
#include "utils.h"

namespace sqcpu = sqaod_cpu;
using namespace sqaod_cuda;

CUDAFormulasBGFuncTest::CUDAFormulasBGFuncTest(void) : MinimalTestSuite("CUDAFormulasBGFuncTest") {
}


CUDAFormulasBGFuncTest::~CUDAFormulasBGFuncTest(void) {
}


void CUDAFormulasBGFuncTest::setUp() {
    device_.initialize();
}

void CUDAFormulasBGFuncTest::tearDown() {
    device_.finalize();
}
    
void CUDAFormulasBGFuncTest::run(std::ostream &ostm) {
    tests<double>();
    tests<float>();
}

template<class real>
void CUDAFormulasBGFuncTest::tests() {

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

    typedef sqcpu::BGFuncs<real> BGF;
    DeviceBipartiteGraphFormulas<real> devFuncs;
    devFuncs.assignDevice(device_);

    const int N0 = 100;
    const int N1 = 120;
    const int m = 110;

    testcase("calculate_E") {
        HostMatrix W = testMat<real>(sq::Dim(N1, N0));
        HostVector b0 = testVec<real>(N0);
        HostVector b1 = testVec<real>(N1);
        HostVector x0 = randomizeBits<real>(N0);
        HostVector x1 = randomizeBits<real>(N1);
        real E;
        BGF::calculate_E(&E, b0, b1, W, x0, x1);

        DeviceMatrix dW;
        DeviceVector db0, db1;
        DeviceVector dx0, dx1;
        DeviceScalar dE;
        devCopy(&dW, W);
        devCopy(&db0, b0);
        devCopy(&db1, b1);
        devCopy(&dx0, x0);
        devCopy(&dx1, x1);
        devFuncs.calculate_E(&dE, db0, db1, dW, dx0, dx1);
        device_.synchronize();
        TEST_ASSERT(dE == E);
    }


    testcase("calculate_E batched") {
        HostMatrix W = testMat<real>(sq::Dim(N1, N0));
        HostVector b0 = testVec<real>(N0);
        HostVector b1 = testVec<real>(N1);
        HostMatrix x0 = randomizeBits<real>(sq::Dim(m, N0));
        HostMatrix x1 = randomizeBits<real>(sq::Dim(m, N1));
        HostVector E;
        BGF::calculate_E(&E, b0, b1, W, x0, x1);

        DeviceMatrix dW;
        DeviceVector db0, db1;
        DeviceMatrix dx0, dx1;
        DeviceVector dE;
        devCopy(&dW, W);
        devCopy(&db0, b0);
        devCopy(&db1, b1);
        devCopy(&dx0, x0);
        devCopy(&dx1, x1);
        devFuncs.calculate_E(&dE, db0, db1, dW, dx0, dx1);
        device_.synchronize();
        TEST_ASSERT(dE == E);
    }

    testcase("calulcate_hJc") {
        HostMatrix W = testMat<real>(sq::Dim(N1, N0));
        HostVector b0 = testVec<real>(N0);
        HostVector b1 = testVec<real>(N1);

        HostVector h0, h1;
        HostMatrix J;
        real c;
        BGF::calculateHamiltonian(&h0, &h1, &J, &c, b0, b1, W);

        DeviceMatrix dW;
        DeviceVector db0, db1;
        devCopy(&dW, W);
        devCopy(&db0, b0);
        devCopy(&db1, b1);
        
        DeviceMatrix dJ;
        DeviceVector dh0, dh1;
        DeviceScalar dc;
        devFuncs.calculateHamiltonian(&dh0, &dh1, &dJ, &dc, db0, db1, dW);
        
        TEST_ASSERT(dh0 == h0);
        TEST_ASSERT(dh1 == h1);
        TEST_ASSERT(dJ == J);
        TEST_ASSERT(dc == c);
    }


    testcase("calulcate_E from hJc") {
        HostMatrix W = testMat<real>(sq::Dim(N1, N0));
        HostVector b0 = testVec<real>(N0);
        HostVector b1 = testVec<real>(N1);

        HostVector h0, h1;
        HostMatrix J;
        real c, E;
        BGF::calculateHamiltonian(&h0, &h1, &J, &c, b0, b1, W);

        HostVector q0 = randomizeBits<real>(N0);
        HostVector q1 = randomizeBits<real>(N1);
        BGF::calculate_E(&E, h0, h1, J, c, q0, q1);

        DeviceMatrix dW;
        DeviceVector db0, db1;
        devCopy(&dW, W);
        devCopy(&db0, b0);
        devCopy(&db1, b1);
        
        DeviceMatrix dJ;
        DeviceVector dh0, dh1;
        DeviceScalar dc;
        devFuncs.calculateHamiltonian(&dh0, &dh1, &dJ, &dc, db0, db1, dW);

        DeviceVector dq0, dq1;
        DeviceScalar dE;
        devCopy(&dq0, q0);
        devCopy(&dq1, q1);
        devFuncs.calculate_E(&dE, dh0, dh1, dJ, dc, dq0, dq1);

        TEST_ASSERT(dE == E);
    }

    testcase("calulcate_E from hJc batched") {
        HostMatrix W = testMat<real>(sq::Dim(N1, N0));
        HostVector b0 = testVec<real>(N0);
        HostVector b1 = testVec<real>(N1);

        HostVector h0, h1;
        HostMatrix J;
        real c;
        HostVector E;
        BGF::calculateHamiltonian(&h0, &h1, &J, &c, b0, b1, W);

        HostMatrix q0 = randomizeBits<real>(sq::Dim(m, N0));
        HostMatrix q1 = randomizeBits<real>(sq::Dim(m, N1));
        BGF::calculate_E(&E, h0, h1, J, c, q0, q1);

        DeviceMatrix dW;
        DeviceVector db0, db1;
        devCopy(&dW, W);
        devCopy(&db0, b0);
        devCopy(&db1, b1);
        
        DeviceMatrix dJ;
        DeviceVector dh0, dh1;
        DeviceScalar dc;
        devFuncs.calculateHamiltonian(&dh0, &dh1, &dJ, &dc, db0, db1, dW);

        DeviceMatrix dq0, dq1;
        DeviceVector dE;
        devCopy(&dq0, q0);
        devCopy(&dq1, q1);
        devFuncs.calculate_E(&dE, dh0, dh1, dJ, dc, dq0, dq1);

        TEST_ASSERT(dE == E);
    }
}
