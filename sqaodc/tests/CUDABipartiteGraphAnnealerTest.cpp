#include "CUDABipartiteGraphAnnealerTest.h"
#include "utils.h"

namespace sqcu = sqaod_cuda;

CUDABipartiteGraphAnnealerTest::CUDABipartiteGraphAnnealerTest(void) : MinimalTestSuite("CUDABipartiteGraphAnnealerTest") {
}


CUDABipartiteGraphAnnealerTest::~CUDABipartiteGraphAnnealerTest(void) {
    
}

void CUDABipartiteGraphAnnealerTest::setUp() {
    device_.useManagedMemory(true);
    device_.initialize();
}

void CUDABipartiteGraphAnnealerTest::tearDown() {
    device_.finalize();
}
    
void CUDABipartiteGraphAnnealerTest::run(std::ostream &ostm) {
    int N0 = 100, N1 = 50;
    int m = N0 + N1;

    sqcu::CUDABipartiteGraphAnnealer<float> annealer;
    sq::MatrixType<float> W = testMat<float>(sq::Dim(N1, N0));
    sq::VectorType<float> b0 = testVec<float>(N0);
    sq::VectorType<float> b1 = testVec<float>(N1);
    annealer.assignDevice(device_);
    annealer.setQUBO(b0, b1, W);
    annealer.setPreference(sq::pnNumTrotters, m);

    testcase("set_q() N") {
        annealer.prepare();
        
        sq::BitSetPair bsetin = createRandomizedSpinSetPair(N0, N1);
        annealer.set_q(bsetin);
        sq::BitSetPairArray bsetout = annealer.get_q();
        
        TEST_ASSERT(compareSolutions(bsetout, bsetin));
    }

    testcase("set_q() N * 2") {
        annealer.prepare();
        
        sq::BitSetPair bsetin;
        sq::BitSetPairArray bsetout;
        bsetin = createRandomizedSpinSetPair(N0, N1);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        bsetin = createRandomizedSpinSetPair(N0, N1);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        
        TEST_ASSERT(compareSolutions(bsetout, bsetin));
    }

    testcase("set_q() m x N") {
        sq::BitSetPairArray bsetin, bsetout;
        bsetin = createRandomizedSpinSetPairArray(N0, N1, m);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

    testcase("set_q() m x N x 2") {
        sq::BitSetPairArray bsetin, bsetout;
        bsetin = createRandomizedSpinSetPairArray(N0, N1, m);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        bsetin = createRandomizedSpinSetPairArray(N0, N1, m);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }
}
