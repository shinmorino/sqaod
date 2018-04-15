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
        
        sq::BitSet bset0 = randomizeBits<char>(N0);
        sq::BitSet bset1 = randomizeBits<char>(N1);
        sq::BitSetPair bsetPair;
        bsetPair.bits0 = sq::x_to_q<char>(bset0);
        bsetPair.bits1 = sq::x_to_q<char>(bset1);
        
        annealer.setPreference(sq::pnNumTrotters, m);
        annealer.prepare();
        annealer.set_q(bsetPair);
        
        bool ok = true;
        sq::BitSetPairArray bsetarr = annealer.get_q();
        for (int idx = 0; idx < m; ++idx) {
            ok &= bsetarr[idx].bits0 == bsetPair.bits0;
            ok &= bsetarr[idx].bits1 == bsetPair.bits1;
        }
        TEST_ASSERT(ok);
    }

    testcase("set_q() m x N") {
        sqcpu::CPUBipartiteGraphAnnealer<float> annealer;
        sq::MatrixType<float> W = testMat<float>(sq::Dim(N1, N0));
        sq::VectorType<float> b0 = testVec<float>(N0);
        sq::VectorType<float> b1 = testVec<float>(N1);
        annealer.setQUBO(b0, b1, W);

        sq::BitSetPairArray bsetin;
        for (int idx = 0; idx < m; ++idx) {
            sq::BitSet bset0 = randomizeBits<char>(N0);
            sq::BitSet bset1 = randomizeBits<char>(N1);
            sq::BitSetPair bsetPair;
            bsetPair.bits0 = sq::x_to_q<char>(bset0);
            bsetPair.bits1 = sq::x_to_q<char>(bset1);
            bsetin.pushBack(bsetPair);
        }
        
        annealer.setPreference(sq::pnNumTrotters, m);
        annealer.set_q(bsetin);
        
        bool ok = true;
        sq::BitSetPairArray bsetout = annealer.get_q();
        if (bsetout.size() == m) {
            for (int idx = 0; idx < m; ++idx) {
                const sq::BitSetPair &in = bsetin[idx];
                const sq::BitSetPair &out = bsetout[idx];
                if ((out.bits0.size == N0) && (out.bits1.size == N1)) {
                    ok &= out.bits0 == in.bits0;
                    ok &= out.bits1 == in.bits1;
                }
                else {
                    ok = false;
                    break;
                }
            }
        }
        else {
            ok = false;
        }
        TEST_ASSERT(ok);
    }
}
