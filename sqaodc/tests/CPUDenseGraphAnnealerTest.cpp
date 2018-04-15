#include "CPUDenseGraphAnnealerTest.h"
#include <utils.h>

namespace sqcpu = sqaod_cpu;

CPUDenseGraphAnnealerTest::CPUDenseGraphAnnealerTest(void) : MinimalTestSuite("CPUDenseGraphAnnealerTest") {
}


CPUDenseGraphAnnealerTest::~CPUDenseGraphAnnealerTest(void) {
    
}


void CPUDenseGraphAnnealerTest::setUp() {
}

void CPUDenseGraphAnnealerTest::tearDown() {
}
    
void CPUDenseGraphAnnealerTest::run(std::ostream &ostm) {
    int N = 100;
    int m = N;
    sqcpu::CPUDenseGraphAnnealer<float> annealer;
    sq::MatrixType<float> W = testMatSymmetric<float>(N);
    annealer.setQUBO(W);
    annealer.setPreference(sq::pnNumTrotters, m);

    testcase("set_q(), N") {
        annealer.prepare();
        
        sq::BitSet bsetin = createRandomizedSpinSet(N);
        annealer.set_q(bsetin);
        sq::BitSetArray bsetout = annealer.get_q();
        
        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

    testcase("set_q(), N x 2") {
        sq::BitSet bsetin;
        sq::BitSetArray bsetout;
        bsetin = createRandomizedSpinSet(N);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();

        bsetin = createRandomizedSpinSet(N);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        
        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

    testcase("set_q(), N x m") {
        sq::BitSetArray bsetin, bsetout;
        bsetin = createRandomizedSpinSetArray(N, m);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();

        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

    testcase("set_q(), m x 2") {
        sq::BitSetArray bsetin, bsetout;
        bsetin = createRandomizedSpinSetArray(N, m);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();

        bsetin = createRandomizedSpinSetArray(N, m);
        annealer.set_q(bsetin);
        bsetout = annealer.get_q();
        
        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

}
