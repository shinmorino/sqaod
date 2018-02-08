#include "DeviceRandomTest.h"
#include <cuda/DeviceRandom.h>
#include <vector>

using namespace sqaod_cuda;

DeviceRandomTest::DeviceRandomTest(void) : MinimalTestSuite("DeviceRandomTest") {
}

DeviceRandomTest::~DeviceRandomTest(void) {
}


void DeviceRandomTest::setUp() {
    device_.initialize();
}

void DeviceRandomTest::tearDown() {
    device_.finalize();
}
    
void DeviceRandomTest::run(std::ostream &ostm) {
    Device device;

    device.initialize();
    auto *alloc = device.objectAllocator();

    testcase("DeviceRandom") {
        const int nRands = 1 << 20;
        DeviceRandom devRand;
        devRand.assignDevice(device_);
        devRand.setRequiredSize(nRands);
        devRand.seed(0);
        TEST_ASSERT(devRand.getNRands() == 0);
        devRand.generate();
        TEST_ASSERT(nRands <= devRand.getNRands());

        sqaod::IdxType offset;
        sqaod::SizeType posToWrap;
        const unsigned int *d_rand = devRand.get(nRands, &offset, &posToWrap);
        TEST_ASSERT(offset == 0);
        d_rand = devRand.get(nRands, &offset, &posToWrap);
        TEST_ASSERT(offset == nRands);

        devRand.synchronize();
        for (int idx = 0; idx < 100; ++idx) {
            d_rand = devRand.get(nRands, &offset, &posToWrap);
            devRand.synchronize();
        }
        TEST_ASSERT(offset == (nRands * 101) % posToWrap);
    }

    device.synchronize();
    device.finalize();
}
