#include "DeviceTest.h"
#include <vector>

DeviceTest::DeviceTest(void) : MinimalTestSuite("DeviceTest") {
}


DeviceTest::~DeviceTest(void) {
}


void DeviceTest::setUp() {
}

void DeviceTest::tearDown() {
}
    
void DeviceTest::run(std::ostream &ostm) {
    testcase("DevInit/Fin") {
        try {
            Device device;
            device.initialize(0);
            device.finalize();
            TEST_SUCCESS;
        }
        catch (...) {
            TEST_FAIL;
        }
    }

    testcase("device alloc/dealloc") {
        Device device;
        device.initialize(0);
        auto *alloc = device.objectAllocator<float>();

        std::vector<void*> pvlist;
        for (int size = 4; size < (1 << 20); size *= 2) {
            for (int idx = 0; idx < 100; ++idx) {
                void *pv = alloc->allocate(size);
                pvlist.push_back(pv);
            }
        }
        for (size_t idx = 0; idx < pvlist.size(); ++idx)
            alloc->deallocate(pvlist[idx]);
        pvlist.clear();
        device.finalize();
        TEST_SUCCESS;
    }

}

template<class real>
void DeviceTest::tests() {
    Device device;

    device.initialize()
    auto *alloc = device_.objectAllocator<real>();

    testcase("matrix alloc/dealloc") {
        DeviceMatrixType<real> mat;
        alloc->allocate(&mat, 10, 10);
        TEST_ASSERT(mat.dim() == sqaod::Dim(10, 10));
        TEST_ASSERT(mat.d_data != NULL);
        alloc->deallocate(mat);
    }

    testcase("vector alloc/dealloc") {
        DeviceVectorType<real> vec;
        alloc->allocate(&vec, 10);
        TEST_ASSERT(vec.size == 10);
        TEST_ASSERT(vec.d_data != NULL);
        alloc->deallocate(vec);
    }

    testcase("scalar alloc/dealloc") {
        DeviceScalarType<real> sc;
        alloc->allocate(&sc);
        TEST_ASSERT(sc.d_data != NULL);
        alloc->deallocate(sc);
    }

    DeviceStream *defStream = device_.defaultStream();

    testcase("tmp object alloc/dealloc") {
        DeviceMatrixType<real> *tmpMat = defStream->tempDeviceMatrix<real>(10, 10);
        DeviceVectorType<real> *tmpVec = defStream->tempDeviceVector<real>(10);
        DeviceScalarType<real> *tmpSc = defStream->tempDeviceScalar<real>();
        defStream->synchronize();
        TEST_ASSERT(true);
    }

    device.synchronize();
    device.finalize();
}
