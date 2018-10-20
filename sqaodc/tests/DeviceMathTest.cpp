#include "DeviceMathTest.h"
#include <cuda/DeviceMath.h>
#include <cuda/DeviceCopy.h>
#include <cpu/SharedFormulas.h>
#include <common/EigenBridge.h>
#include "utils.h"

namespace sqcpu = sqaod_cpu;
namespace sqcu = sqaod_cuda;
namespace sq = sqaod;

DeviceMathTest::DeviceMathTest(void) : MinimalTestSuite("DeviceMathTest") {
}

DeviceMathTest::~DeviceMathTest(void) {
}

void DeviceMathTest::setUp() {
    device_.useManagedMemory(true);
    device_.initialize(0);
}

void DeviceMathTest::tearDown() {
    device_.finalize();
}

void DeviceMathTest::run(std::ostream &ostm) {

    sq::Dim dims[] = { {100, 100} , {32, 33}, {33, 32}};
    // sq::Dim dim(8, 5), sq::Dim dim(5, 8) sq::Dim dim(5, 5);
    for (int idx = 0; idx < 3; ++idx) {
        tests<double>(dims[idx]);
        tests<float>(dims[idx]);
    }
}

template<class real>
void DeviceMathTest::tests(const sqaod::Dim &dim) {

    sqcu::DeviceMathType<real> devMath(device_);
    sqcu::DeviceCopy devCopy(device_);

    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef sq::EigenMatrixType<real> EigenMatrix;
    typedef sq::EigenRowVectorType<real> EigenRowVector;
    typedef sq::EigenColumnVectorType<real> EigenColumnVector;
    typedef sqcu::DeviceMatrixType<real> DeviceMatrix;
    typedef sqcu::DeviceVectorType<real> DeviceVector;
    typedef sqcu::DeviceScalarType<real> DeviceScalar;


    auto *alloc = device_.objectAllocator();
    
    DeviceMatrix dA, dB, dC, dD;
    DeviceVector dx, dy, dz;
    DeviceScalar da, db, dc;

    testcase("test zeros/eye") {
        sq::Dim dim1(dim.rows, dim.rows);
        alloc->allocate(&dA, dim1);
        devCopy.broadcast(&dA, (real)0.); /* create zero matrix */
        device_.synchronize();
        TEST_ASSERT(dA == HostMatrix::zeros(dim1));

        devMath.broadcastToDiagonal(&dA, real(1.));
        device_.synchronize();
        TEST_ASSERT(dA == HostMatrix::eye(dim.rows));
    }

    testcase("mat scale/sum") {
        HostMatrix hMat = testMatBalanced<real>(dim);
        devCopy(&dA, hMat);
        devMath.scale(&dB, 10., dA);
        device_.synchronize();
        hMat *= (real)10.;
        TEST_ASSERT(dB == hMat);

        devMath.sum(&da, real(3.), dB);
        device_.synchronize();
        TEST_ASSERT(da == real(3. * hMat.sum()));
        /* mulAddAssign */
        devMath.sum(&da, real(3.), dB, 1.);
        device_.synchronize();
        TEST_ASSERT(da == real(6. * hMat.sum()));
    }

    testcase("vec scale/sum") {
        HostVector hVec = testVec<real>(dim.cols);
        devCopy(&dx, hVec);

        devMath.scale(&dy, 10., dx);
        device_.synchronize();
        hVec *= (real)10.;
        TEST_ASSERT(dy == hVec);

        devMath.scale(&dy, 10., dx, 1.);
        device_.synchronize();
        hVec *= (real)2.;
        TEST_ASSERT(dy == hVec);

        devMath.sum(&da, real(3.), dy);
        device_.synchronize();
        TEST_ASSERT(da == real(3. * hVec.sum()));
        /* mulAddAssign */
        devMath.sum(&da, real(3.), dy, 2.);
        TEST_ASSERT(da == real((3. + 6.) * hVec.sum()));
    }

    testcase("scalar scale") {
        real hsc = 35.;
        devCopy(&da, hsc);

        devMath.scale(&db, 10., da);
        device_.synchronize();
        TEST_ASSERT(db == real(hsc * 10.));

        devMath.scale(&db, 10., da, 2.);
        device_.synchronize();
        TEST_ASSERT(db == real(hsc * 30.));
    }

    testcase("vector scale broadcast") {
        alloc->allocate(&dx, dim.rows);
        alloc->allocate(&dy, dim.cols);
        alloc->allocate(&da);

        /* initialize */
        devCopy(&da, (real)0.);
        devMath.scaleBroadcast(&dx, 1., da);
        HostVector x = HostVector::zeros(dim.rows);
        device_.synchronize();
        TEST_ASSERT(dx == x);

        devCopy(&da, (real)1.);
        devMath.scaleBroadcast(&dx, 2., da);
        x = HostVector::ones(dim.rows);
        x *= (real)2.;
        TEST_ASSERT(dx == x);

        devMath.scaleBroadcast(&dx, 2., da, 1.);
        x *= (real)2.;
        TEST_ASSERT(dx == x);
    }

    testcase("matrix scale broadcast") {
        alloc->allocate(&dA, dim);
        alloc->allocate(&dx, dim.cols);
        alloc->allocate(&dy, dim.rows);

        HostMatrix hmat(dim);
        HostVector x = testVec<real>(dim.cols);
        HostVector y = testVec<real>(dim.rows);
        devCopy(&dx, x);
        devCopy(&dy, y);

        devMath.scaleBroadcast(&dA, 1., dx, sqcu::opRowwise);
        sq::mapTo(hmat).rowwise() = sq::mapToRowVector(x);
        device_.synchronize();
        TEST_ASSERT(dA == hmat);

        devMath.scaleBroadcast(&dA, 2., dx, sqcu::opRowwise, 2.);
        sq::mapTo(hmat).rowwise() = 4. * sq::mapToRowVector(x);
        device_.synchronize();
        TEST_ASSERT(dA == hmat);

        devMath.scaleBroadcast(&dA, 1., dy, sqcu::opColwise);
        sq::mapTo(hmat).colwise() = sq::mapToColumnVector(y);
        device_.synchronize();
        TEST_ASSERT(dA == hmat);

        devMath.scaleBroadcast(&dA, 2., dy, sqcu::opColwise, 2.);
        sq::mapTo(hmat).colwise() = 4. * sq::mapToColumnVector(y);
        device_.synchronize();
        TEST_ASSERT(dA == hmat);
    }


    testcase("matrix sum") {
        HostMatrix hmat = testMat<real>(dim);
        alloc->allocate(&dA, dim);
        alloc->allocate(&da);

        devCopy(&dA, hmat);
        devMath.sum(&da, 1., dA);
        device_.synchronize();
        TEST_ASSERT(da == hmat.sum());

        devMath.sum(&da, 2., dA, 2.);
        device_.synchronize();
        TEST_ASSERT(da == real(4. * hmat.sum()));
    }

    testcase("vector sum") {
        HostVector hvec = testVec<real>(dim.rows);
        alloc->allocate(&dx, dim.rows);
        alloc->allocate(&da);

        devCopy(&dx, hvec);
        devMath.sum(&da, 1., dx);
        device_.synchronize();
        TEST_ASSERT(da == hvec.sum());

        devMath.sum(&da, 2., dx, 2.);
        device_.synchronize();
        TEST_ASSERT(da == real(4. * hvec.sum()));
    }

    testcase("diagonal sum") {
        HostMatrix hmat = testMat<real>(dim);
        alloc->allocate(&dA, dim);
        alloc->allocate(&da);

        devCopy(&dA, hmat);
        devMath.sumDiagonal(&da, dA);
        device_.synchronize();
        TEST_ASSERT(da == sq::mapTo(hmat).diagonal().sum());
    }
    testcase("sum batched") {
        alloc->allocate(&dA, dim);
        alloc->allocate(&dx, dim.rows); /* col vector */
        alloc->allocate(&dy, dim.cols); /* row vector */

        HostMatrix hmat = testMat<real>(dim);
        HostVector x = testVec<real>(dim.rows); /* col vector */
        HostVector y = testVec<real>(dim.cols); /* row vector */
        devCopy(&dA, hmat);
        devCopy(&dx, x);
        devCopy(&dy, y);

        devMath.sumBatched(&dx, 2., dA, sqcu::opRowwise);

        HostVector hvec(dim.rows);
        mapToRowVector(hvec) = 2. * sq::mapTo(hmat).rowwise().sum();
        device_.synchronize();
        TEST_ASSERT(dx == hvec);

        devMath.sumBatched(&dy, 2., dA, sqcu::opColwise);
        hvec.resize(dim.cols);
        mapToColumnVector(hvec) = 2. * sq::mapTo(hmat).colwise().sum();
        device_.synchronize();
        TEST_ASSERT(dy == hvec);
    }

    testcase("dot") {
        HostVector x = testVec<real>(dim.rows);
        HostVector y = testVec<real>(dim.rows);
        alloc->allocate(&dx, dim.rows);
        alloc->allocate(&dy, dim.rows);
        alloc->allocate(&da);

        devCopy(&dx, x);
        devCopy(&dy, y);
        devMath.dot(&da, 2., dx, dy);
        device_.synchronize();
        real product = mapToColumnVector(x).dot(mapToColumnVector(y));
        TEST_ASSERT(da == real(2. * product));

        devMath.dot(&da, 2., dx, dy, 1.);
        device_.synchronize();
        TEST_ASSERT(da == real(product * 4.));
    }

    testcase("dot batched") {
        alloc->allocate(&dA, dim);
        alloc->allocate(&dB, dim);
        alloc->allocate(&dx, dim.rows); /* col vector */
        alloc->allocate(&dy, dim.cols); /* row vector */

        HostMatrix A = testMatBalanced<real>(dim);
        HostMatrix B = testMatBalanced<real>(dim);
        devCopy(&dA, A);
        devCopy(&dB, B);
        devMath.dotBatched(&dx, 2., dA, sqcu::opNone, dB, sqcu::opNone);

        EigenMatrix eAB = sq::mapTo(A).array() * sq::mapTo(B).array();
        EigenRowVector vec = 2. * eAB.rowwise().sum();
        device_.synchronize();

        TEST_ASSERT(dx == sq::mapFrom(vec));

        devMath.dotBatched(&dy, 2., dA, sqcu::opTranspose, dB, sqcu::opTranspose);
        vec = 2. * eAB.colwise().sum();
        device_.synchronize();

        TEST_ASSERT(dy == sq::mapFrom(vec));
    }

    testcase("transpose") {
        HostMatrix hMat = testMat<real>(dim);
        devCopy(&dA, hMat);
        devMath.transpose(&dB, dA);
        device_.synchronize();
        HostMatrix hTrans(hMat.dim().transpose());
        sq::mapTo(hTrans) = sq::mapTo(hMat).transpose();
        TEST_ASSERT(dB == hTrans);
    }

#if 0
    testcase("symmetrize") {
        HostMatrix hMat = testMat<real>(sq::Dim(dim.rows, dim.rows));
        devCopy(&dA, hMat);
        devMath.symmetrize(&dB, dA);
        device_.synchronize();
        HostMatrix hSym = sqcpu::symmetrize(hMat);
        TEST_ASSERT(dB == hSym);
    }
#endif

    testcase("mvProduct") {
        HostMatrix A = testMatBalanced<real>(dim);
        HostVector x = testVecBalanced<real>(dim.cols);
        devCopy(&dA, A);
        devCopy(&dx, x);
        devMath.mvProduct(&dy, 0.5, dA, sqcu::opNone, dx);
        EigenRowVector y = 0.5 * sq::mapTo(A) * sq::mapToColumnVector(x);
        device_.synchronize();
        TEST_ASSERT(dy == sq::mapFrom(y));

        HostMatrix B = testMatBalanced<real>(dim.transpose());
        devCopy(&dB, B);
        devMath.mvProduct(&dy, 0.25, dB, sqcu::opTranspose, dx);
        y = sq::mapTo(B).transpose() * sq::mapToColumnVector(x);
        y *= 0.25;
        device_.synchronize();
        TEST_ASSERT(dy == sq::mapFrom(y));
    }

    testcase("mmProduct") {
        HostMatrix A = testMatBalanced<real>(dim);
        HostMatrix B = testMatBalanced<real>(dim.transpose());
        EigenMatrix C = 0.5 * sq::mapTo(A) * sq::mapTo(B);

        devCopy(&dA, A);
        devCopy(&dB, B);
        devMath.mmProduct(&dC, 0.5, dA, sqcu::opNone, dB, sqcu::opNone);
        device_.synchronize();
        TEST_ASSERT(allclose(dC, sq::mapFrom(C), epusiron<real>()));

        HostMatrix At(dim.transpose());
        HostMatrix Bt(dim);
        sq::mapTo(At) = sq::mapTo(A).transpose();
        sq::mapTo(Bt) = sq::mapTo(B).transpose();
        alloc->deallocate(dA);
        alloc->deallocate(dB);
        alloc->deallocate(dC);
        devCopy(&dA, At);
        devCopy(&dB, Bt);
        devMath.mmProduct(&dC, 0.5, dA, sqcu::opTranspose, dB, sqcu::opTranspose);
        device_.synchronize();
        TEST_ASSERT(allclose(dC, sq::mapFrom(C), epusiron<real>()));
    }

    testcase("vmvProduct") {
        HostMatrix A = testMatBalanced<real>(dim);
        HostVector x = testVecBalanced<real>(dim.cols);
        HostVector y = testVecBalanced<real>(dim.rows);
        real product = mapToRowVector(y) * sq::mapTo(A) * mapToColumnVector(x);

        devCopy(&dA, A);
        devCopy(&dx, x);
        devCopy(&dy, y);
        devMath.vmvProduct(&da, 2., dy, dA, dx);
        device_.synchronize();
        TEST_ASSERT(da == real(2. * product));
    }

    testcase("vmvProductBatched") {
        sqaod::Dim dimSq(dim.cols, dim.cols);
        HostMatrix A = testMatBalanced<real>(dimSq);
        HostMatrix X = testMatBalanced<real>(dim);
        HostMatrix Y = testMatBalanced<real>(dim);

        EigenMatrix AX = sq::mapTo(A) * sq::mapTo(X).transpose();
        EigenMatrix AXY = sq::mapTo(Y).array() * AX.transpose().array();
        EigenColumnVector z = 1. * AXY.rowwise().sum();

        devCopy(&dA, A);
        devCopy(&dB, X);
        devCopy(&dC, Y);
        devMath.vmvProductBatched(&dx, 2., dC, dA, dB);
        z *= 2.;
        device_.synchronize();
        TEST_ASSERT(dx == sq::mapFrom(z));
    }

    testcase("min matrix") {
        HostMatrix A = testMatBalanced<real>(dim);
        real vMin = sq::mapTo(A).minCoeff();
        devCopy(&dA, A);
        devMath.min(&da, dA);
        device_.synchronize();
        TEST_ASSERT(da == vMin);
    }

    testcase("min") {
        HostVector x = testVecBalanced<real>(dim.rows);
        real vMin = sq::mapToRowVector(x).minCoeff();
        devCopy(&dx, x);
        devMath.min(&da, dx);
        device_.synchronize();
        TEST_ASSERT(da == vMin);
    }

    alloc->deallocate(dA);
    alloc->deallocate(dB);
    alloc->deallocate(dC);
    alloc->deallocate(dD);
    alloc->deallocate(dx);
    alloc->deallocate(dy);
    alloc->deallocate(da);
}
