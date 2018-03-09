#include "CUDADenseGraphAnnealerTest.h"
#include <cuda_runtime.h>
#include <cuda/DeviceRandomBuffer.h>
#include <cuda/DeviceStream.h>
#include <cuda/CUDADenseGraphAnnealer.h>
#include <cuda/DeviceFormulas.h>
#include "utils.h"
#include <cpu/CPUFormulas.h>
#include <common/EigenBridge.h>

using namespace sqaod_cuda;
namespace sqcpu = sqaod_cpu;


CUDADenseGraphAnnealerTest::CUDADenseGraphAnnealerTest(void) : MinimalTestSuite("CUDADenseGraphAnnealerTest")
{
}


CUDADenseGraphAnnealerTest::~CUDADenseGraphAnnealerTest(void)
{
}


void CUDADenseGraphAnnealerTest::setUp() {
    device_.useManagedMemory(true);
    device_.initialize();
}

void CUDADenseGraphAnnealerTest::tearDown() {
    device_.finalize();
}

void CUDADenseGraphAnnealerTest::run(std::ostream &ostm) {
    test<float>();
    //test<double>();
}

template<class real>
void CUDADenseGraphAnnealerTest::test() {

    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;

    DeviceStream *devStream = device_.defaultStream();
    DeviceObjectAllocator *devAlloc = device_.objectAllocator();
    DeviceCopy devCopy(device_);
    DeviceDenseGraphFormulas<real> dgFuncs(device_);
    int N = 40;
    int m = 20;

    testcase("DeviceRandomBuffer generateFlipPos()") {
        DeviceRandom d_random(device_);
        DeviceRandomBuffer buffer(device_);
        d_random.setRequiredSize(1 << 20);
        d_random.seed(0);
        buffer.generateFlipPositions(d_random, N, m, 8);
        devStream->synchronize();
        TEST_ASSERT(buffer.available(N* m * 2));
        const int *d_flippos = buffer.acquire<int>(N * m * 8);
        bool ok = true;
        for (int idx = 0; idx < 4; ++idx) {
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < m; ++y) {
                    int offset = y % 2;
                    int v = d_flippos[x + y * N];
                    ok &= ((v - offset) % 1) == 0;
                    ok &= ((0 <= v) && (v < N));
                }
            }
            for (int x = 0; x < N; ++x) {
                for (int y = 0; y < m; ++y) {
                    int offset = (y + 1) % 2;
                    int v = d_flippos[x + y * N];
                    ok &= ((v - offset) % 1) == 0;
                    ok &= ((0 <= v) && (v < N));
                }
            }
        }
        TEST_ASSERT(ok);
    }

    testcase("DeviceRandomBuffer reandom<real>") {
        DeviceRandom d_random(device_);
        DeviceRandomBuffer buffer(device_);
        d_random.setRequiredSize(1 << 20);
        d_random.seed(0);
        buffer.generate<real>(d_random, 1 << 20);
        devStream->synchronize();
        TEST_ASSERT(buffer.available(1 << 20));
        const real *d_real = buffer.acquire<real>(1 << 20);
        bool ok = true;
        real sum = real(0.);
        for (int idx = 0; idx < 1 << 20; ++idx) {
            ok &= ((real(0.) <= d_real[idx]) && (d_real[idx] < real(1.)));
            sum += d_real[idx];
        }
        TEST_ASSERT(ok);
        real av = sum / (1 << 20);
        TEST_ASSERT(std::fabs(av - real(0.5)) < real(0.05));
    }

    testcase("DeviceRandomBuffer buffer access") {
        int N = 1 << 20;
        DeviceRandom d_random(device_);
        DeviceRandomBuffer buffer(device_);
        d_random.setRequiredSize(N);
        d_random.seed(0);
        buffer.generate<real>(d_random, N);
        devStream->synchronize();

        bool ok = true;
        for (int idx = 0; idx < N; idx += N / 8) {
            ok &= buffer.available(N / 8);
            buffer.acquire<real>(N / 8);
        }
        ok &= !buffer.available(N / 8);
        TEST_ASSERT(ok);
    }

    testcase("calculate_Jq()") {
        const int N = 40;
        const int m = 20;

        HostMatrix W = testMatSymmetric<real>(N);
        HostMatrix J;
        HostVector Jq(m);
        HostVector h;
        real c;
        sqcpu::DGFuncs<real>::calculate_hJc(&h, &J, &c, W);
        int flippos[m];
        for (int idx = 0; idx < m; ++idx)
            flippos[idx] = (idx * 3) % m;

        HostMatrix q(m, N);
        for (int idx = 0; idx < N * m; ++idx)
            q.data[idx] = real((idx % 2) * 2 - 1);
        for (int idx = 0; idx < m; ++idx) {
            sq::mapToRowVector(Jq)(idx) = sq::mapTo(J).row(flippos[idx]).dot(sq::mapTo(q).row(idx));
#if 0
            real sum = real();
            for (int inner = 0; inner < N; ++inner) {
                printf("%g, ", J(flippos[idx], inner) * q(idx, inner));
            }
#endif
        }

        CUDADenseGraphAnnealer<real> an(device_);
        an.setProblem(W);
        an.setPreference(sq::Preference(sq::pnNumTrotters, m));
        an.seed(0);
        // an.randomize_q();  FIXME: initilization check.
        an.prepare();

        DeviceMatrix d_W, d_J, d_q;
        DeviceVector d_Jq, d_h;
        DeviceScalar d_c;

        int *d_flippos;
        devAlloc->allocate(&d_flippos, m);
        devAlloc->allocate(&d_Jq, m);

        devCopy(&d_W, W);
        dgFuncs.calculate_hJc(&d_h, &d_J, &d_c, d_W);
        devStream->synchronize();

        devCopy(&d_q, q);
        DeviceBitMatrix *d_bitQ = devStream->tempDeviceMatrix<char>(q.dim());
        devCopy.cast(d_bitQ , d_q);

        devCopy(d_flippos, flippos, m);
        an.calculate_Jq(&d_Jq, d_J, *d_bitQ , d_flippos);
        devStream->synchronize();

        TEST_ASSERT(d_Jq == Jq);
#if 0
        std::cout << HostVector(d_J.row(0), N) << std::endl;
        std::cout << d_q << std::endl;
        std::cout << sq::mapTo(J).row(0) << std::endl;
        std::cout << sq::mapTo(q) << std::endl;
#endif
    }

}
