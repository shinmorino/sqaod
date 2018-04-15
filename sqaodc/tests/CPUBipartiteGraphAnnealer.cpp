#include "CPUBipartiteGraphAnnealerTest.h"
#include <utils.h>

namespace sqcpu = sqaod_cpu;

CPUBipartiteGraphAnnealerTest::CPUBipartiteGraphAnnealerTest(void) : MinimalTestSuite("CPUBipartiteGraphAnnealerTest") {
}


CPUBipartiteGraphAnnealerTest::~CPUBipartiteGraphAnnealerTest(void) {
    
}


void CPUBipartiteGraphAnnealerTest::setUp() {
}

void CPUBipartiteGraphAnnealerTest::tearDown() {
}
    
void CPUBipartiteGraphAnnealerTest::run(std::ostream &ostm) {
    int N0 = 100, N1 = 50;
    int m = N0 + N1;

    testcase("set_q() N") {
        sqcpu::CPUBipartiteGraphAnnealer<float> annealer;
        sq::MatrixType<float> W = testMat<float>(sq::Dim(N1, N0));
        sq::VectorType<float> b0 = testVec<float>(N0);
        sq::VectorType<float> b1 = testVec<float>(N1);
        annealer.setQUBO(b0, b1, W);
        annealer.setPreference(sq::pnNumTrotters, m);
        annealer.prepare();
        
        sq::BitSetPair bsetin = createRandomizedSpinSetPair(N0, N1);
        annealer.set_q(bsetin);
        sq::BitSetPairArray bsetout = annealer.get_q();
        
        TEST_ASSERT(compareSolutions(bsetout, bsetin));
    }

    testcase("set_q() N * 2") {
        sqcpu::CPUBipartiteGraphAnnealer<float> annealer;
        sq::MatrixType<float> W = testMat<float>(sq::Dim(N1, N0));
        sq::VectorType<float> b0 = testVec<float>(N0);
        sq::VectorType<float> b1 = testVec<float>(N1);
        annealer.setQUBO(b0, b1, W);
        annealer.setPreference(sq::pnNumTrotters, m);
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
        sqcpu::CPUBipartiteGraphAnnealer<float> annealer;
        sq::MatrixType<float> W = testMat<float>(sq::Dim(N1, N0));
        sq::VectorType<float> b0 = testVec<float>(N0);
        sq::VectorType<float> b1 = testVec<float>(N1);
        annealer.setQUBO(b0, b1, W);

        sq::BitSetPairArray bsetin, bsetout;
        bsetin = createRandomizedSpinSetPairArray(N0, N1, m);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

    testcase("set_q() m x N x 2") {
        sqcpu::CPUBipartiteGraphAnnealer<float> annealer;
        sq::MatrixType<float> W = testMat<float>(sq::Dim(N1, N0));
        sq::VectorType<float> b0 = testVec<float>(N0);
        sq::VectorType<float> b1 = testVec<float>(N1);
        annealer.setQUBO(b0, b1, W);

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
