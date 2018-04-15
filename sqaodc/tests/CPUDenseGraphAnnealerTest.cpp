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

    testcase("set_q(), N") {
        sqcpu::CPUDenseGraphAnnealer<float> annealer;
        sq::MatrixType<float> W = testMatSymmetric<float>(N);
        annealer.setQUBO(W);
        
        sq::BitSet bset = randomizeBits<char>(N);
        bset = sq::x_to_q<char>(bset);
        annealer.setPreference(sq::pnNumTrotters, m);
        annealer.prepare();
        annealer.set_q(bset);
        
        bool ok = true;
        sq::BitSetArray bsetarr = annealer.get_q();
        for (int idx = 0; idx < m; ++idx)
            ok &= bset == bsetarr[idx];
        TEST_ASSERT(ok);
    }

    testcase("set_q(), N x m") {
        sqcpu::CPUDenseGraphAnnealer<float> annealer;
        sq::MatrixType<float> W = testMatSymmetric<float>(N);
        annealer.setQUBO(W);
        
        sq::BitSetArray bsetin;
        for (int idx = 0; idx < m; ++idx) {
            sq::BitSet bset = randomizeBits<char>(N);
            bset = sq::x_to_q<char>(bset);
            bsetin.pushBack(bset);
        }
        annealer.set_q(bsetin);
        sq::BitSetArray bsetout = annealer.get_q();

        bool ok = true;
        if (bsetout.size() == m) {
            for (int idx = 0; idx < m; ++idx) {
                ok &= bsetin[idx] == bsetout[idx];
            }
        }
        else {
            ok = false;
        }
        TEST_ASSERT(ok);
    }
}
