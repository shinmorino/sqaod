#include "DeviceMathTest.h"
#include <cuda/DeviceMath.h>
#include <cuda/DeviceCopy.h>
#include "utils.h"

namespace sq = sqaod;

DeviceMathTest::DeviceMathTest(void) : MinimalTestSuite("DeviceMathTest") {
}

DeviceMathTest::~DeviceMathTest(void) {
}

void DeviceMathTest::setUp() {
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

    DeviceMathType<real> devMath(device_);
    DeviceCopyType<real> devCopy(device_);
    DeviceStream *devStream = device_.defaultStream();

    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef sq::EigenMatrixType<real> EigenMatrix;
    typedef sq::EigenRowVectorType<real> EigenRowVector;
    typedef sq::EigenColumnVectorType<real> EigenColumnVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;


    auto *alloc = device_.objectAllocator<real>();
    
    DeviceMatrix dA, dB, dC, dD;
    DeviceVector dx, dy, dz;
    DeviceScalar da, db, dc;

    testcase("test zeros/eye") {
        sq::Dim dim1(dim.rows, dim.rows);
        alloc->allocate(&dA, dim1);
        devCopy(&dA, 0.); /* create zero matrix */
        device_.synchronize();
        TEST_ASSERT(dA == HostMatrix::zeros(dim1));

        devMath.setToDiagonals(&dA, real(1.));
        device_.synchronize();
        TEST_ASSERT(dA == HostMatrix::eye(dim.rows));
    }

    testcase("mat scale/sum") {
        HostMatrix hMat = testMat<real>(dim);
        devCopy(&dA, hMat);
        devMath.scale(&dB, 10., dA);
        device_.synchronize();
        hMat.map() *= 10.;
        TEST_ASSERT(dB == hMat);

        devMath.sum(&da, real(3.), dB);
        device_.synchronize();
        TEST_ASSERT(da == real(3. * hMat.map().sum()));
        /* mulAddAssign */
        devMath.sum(&da, real(3.), dB, 1.);
        device_.synchronize();
        TEST_ASSERT(da == real(6. * hMat.map().sum()));
    }

    testcase("vec scale/sum") {
        HostVector hVec = testVec<real>(dim.cols);
        devCopy(&dx, hVec);

        devMath.scale(&dy, 10., dx);
        device_.synchronize();
        hVec.mapToRowVector() *= 10.;
        TEST_ASSERT(dy == hVec);

        devMath.scale(&dy, 10., dx, 1.);
        device_.synchronize();
        hVec.mapToRowVector() *= 2.;
        TEST_ASSERT(dy == hVec);

        devMath.sum(&da, real(3.), dy);
        device_.synchronize();
        TEST_ASSERT(da == real(3. * hVec.mapToRowVector().sum()));
        /* mulAddAssign */
        devMath.sum(&da, real(3.), dy, 2.);
        TEST_ASSERT(da == real((3. + 6.) * hVec.mapToRowVector().sum()));
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
        devCopy(&da, 0.);
        devMath.scaleBroadcast(&dx, 1., da);
        HostVector x = HostVector::zeros(dim.rows);
        device_.synchronize();
        TEST_ASSERT(dx == x);

        devCopy(&da, 1.);
        devMath.scaleBroadcast(&dx, 2., da);
        x = HostVector::ones(dim.rows);
        x.mapToRowVector() *= 2.;
        TEST_ASSERT(dx == x);

        devMath.scaleBroadcast(&dx, 2., da, 1.);
        x.mapToRowVector() *= 2.;
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

        devMath.scaleBroadcast(&dA, 1., dx, opRowwise);
        hmat.map().rowwise() = x.mapToRowVector();
        device_.synchronize();
        TEST_ASSERT(dA == hmat);

        devMath.scaleBroadcast(&dA, 2., dx, opRowwise, 2.);
        hmat.map().rowwise() = 4. * x.mapToRowVector();
        device_.synchronize();
        TEST_ASSERT(dA == hmat);

        devMath.scaleBroadcast(&dA, 1., dy, opColwise);
        hmat.map().colwise() = y.mapToColumnVector();
        device_.synchronize();
        TEST_ASSERT(dA == hmat);

        devMath.scaleBroadcast(&dA, 2., dy, opColwise, 2.);
        hmat.map().colwise() = 4. * y.mapToColumnVector();
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
        TEST_ASSERT(da == hmat.map().sum());

        devMath.sum(&da, 2., dA, 2.);
        device_.synchronize();
        TEST_ASSERT(da == real(4. * hmat.map().sum()));
    }

    testcase("vector sum") {
        HostVector hvec = testVec<real>(dim.rows);
        alloc->allocate(&dx, dim.rows);
        alloc->allocate(&da);

        devCopy(&dx, hvec);
        devMath.sum(&da, 1., dx);
        device_.synchronize();
        TEST_ASSERT(da == hvec.mapToRowVector().sum());

        devMath.sum(&da, 2., dx, 2.);
        device_.synchronize();
        TEST_ASSERT(da == real(4. * hvec.mapToRowVector().sum()));
    }

    testcase("diagonal sum") {
        HostMatrix hmat = testMat<real>(dim);
        alloc->allocate(&dA, dim);
        alloc->allocate(&da);

        devCopy(&dA, hmat);
        devMath.sumDiagonals(&da, dA);
        device_.synchronize();
        TEST_ASSERT(da == hmat.map().diagonal().sum());
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

        devMath.sumBatched(&dx, 2., dA, opRowwise);

        HostVector hvec(dim.rows);
        hvec.mapToRowVector() = 2. * hmat.map().rowwise().sum();
        device_.synchronize();
        TEST_ASSERT(dx == hvec);

        devMath.sumBatched(&dy, 2., dA, opColwise);
        hvec.resize(dim.cols);
        hvec.mapToColumnVector() = 2. * hmat.map().colwise().sum();
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
        real product = x.mapToColumnVector().dot(y.mapToColumnVector());
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
        devMath.dotBatched(&dx, 2., dA, opNone, dB, opNone);

        EigenMatrix eAB = A.map().array() * B.map().array();
        EigenRowVector vec = 2. * eAB.rowwise().sum();
        device_.synchronize();

        TEST_ASSERT(dx == HostVector(vec));

        devMath.dotBatched(&dy, 2., dA, opTranspose, dB, opTranspose);
        vec = 2. * eAB.colwise().sum();
        device_.synchronize();

        TEST_ASSERT(dy == HostVector(vec));
    }

    testcase("transpose") {
        HostMatrix hMat = testMat<real>(dim);
        devCopy(&dA, hMat);
        devMath.transpose(&dB, dA);
        device_.synchronize();
        HostMatrix hTrans(hMat.dim().transpose());
        hTrans.map() = hMat.map().transpose();
        TEST_ASSERT(dB == hTrans);
    }

    testcase("mvProduct") {
        HostMatrix A = testMatBalanced<real>(dim);
        HostVector x = testVecBalanced<real>(dim.cols);
        devCopy(&dA, A);
        devCopy(&dx, x);
        devMath.mvProduct(&dy, 0.5, dA, opNone, dx);
        EigenRowVector y = 0.5 * A.map() * x.mapToColumnVector();
        device_.synchronize();
        TEST_ASSERT(dy == HostVector(y));

        HostMatrix B = testMatBalanced<real>(dim.transpose());
        devCopy(&dB, B);
        devMath.mvProduct(&dy, 0.25, dB, opTranspose, dx);
        y = B.map().transpose() * x.mapToColumnVector();
        device_.synchronize();
        TEST_ASSERT(dy == HostVector(0.25 * y));
    }

    testcase("mmProduct") {
        HostMatrix A = testMatBalanced<real>(dim);
        HostMatrix B = testMatBalanced<real>(dim.transpose());
        EigenMatrix C = 0.5 * A.map() * B.map();

        devCopy(&dA, A);
        devCopy(&dB, B);
        devMath.mmProduct(&dC, 0.5, dA, opNone, dB, opNone);
        device_.synchronize();
        TEST_ASSERT(dC == HostMatrix(C));

        HostMatrix At(dim.transpose());
        HostMatrix Bt(dim);
        At.map() = A.map().transpose();
        Bt.map() = B.map().transpose();
        alloc->deallocate(dA);
        alloc->deallocate(dB);
        alloc->deallocate(dC);
        devCopy(&dA, At);
        devCopy(&dB, Bt);
        devMath.mmProduct(&dC, 0.5, dA, opTranspose, dB, opTranspose);
        device_.synchronize();
        TEST_ASSERT(dC == HostMatrix(C));
    }

    testcase("vmvProduct") {
        HostMatrix A = testMatBalanced<real>(dim);
        HostVector x = testVecBalanced<real>(dim.cols);
        HostVector y = testVecBalanced<real>(dim.rows);
        real product = y.mapToRowVector() * A.map() * x.mapToColumnVector();

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

        EigenMatrix AX = A.map() * X.map().transpose();
        EigenMatrix AXY = Y.map().array() * AX.transpose().array();
        EigenColumnVector z = 1. * AXY.rowwise().sum();

        devCopy(&dA, A);
        devCopy(&dB, X);
        devCopy(&dC, Y);
        devMath.vmvProductBatched(&dx, 2., dC, dA, dB);
        device_.synchronize();
        TEST_ASSERT(dx == HostVector(2. * z));
    }

    testcase("min") {
        HostMatrix A = testMatBalanced<real>(dim);
        real vMin = A.map().minCoeff();
        devCopy(&dA, A);
        devMath.min(&da, dA);
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
