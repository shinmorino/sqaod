#include "CPUDenseGraphAnnealerTest.h"
#include <cpu/SharedFormulas.h>
#include "utils.h"

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

    testcase("symmetrize") {
        sq::MatrixType<float> W = sq::MatrixType<float>::zeros(N, N);
        for (sqaod::SizeType iRow = 0; iRow < N; ++iRow) {
            for (sqaod::SizeType iCol = iRow; iCol < N; ++iCol) {
                W(iRow, iCol) = float(iRow * 10 + iCol);
            }
        }
        sq::MatrixType<float> Wsym0 = W;
        for (sqaod::SizeType iRow = 0; iRow < N; ++iRow) {
            for (sqaod::SizeType iCol = iRow + 1; iCol < N; ++iCol) {
                Wsym0(iCol, iRow) = Wsym0(iRow, iCol) = W(iRow, iCol) / 2.;
            }
        }

        sq::MatrixType<float> Wsym1 = sqcpu::symmetrize(W);
        TEST_ASSERT(sq::isSymmetric(Wsym1));
        TEST_ASSERT(Wsym0 == Wsym1);
    }

    testcase("symmetrize_1") {
        sq::MatrixType<float> W = createRandomSymmetricMatrix<float>(N);
        sq::MatrixType<float> Wsym = sqcpu::symmetrize(W);
        TEST_ASSERT(sq::isSymmetric(Wsym));
    }

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
        annealer.prepare();

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
        annealer.set_qset(bsetin);
        bsetout = annealer.get_q();

        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

    testcase("set_q(), m x 2") {
        sq::BitSetArray bsetin, bsetout;
        bsetin = createRandomizedSpinSetArray(N, m);
        annealer.set_qset(bsetin);
        bsetout = annealer.get_q();

        bsetin = createRandomizedSpinSetArray(N, m);
        annealer.set_qset(bsetin);
        bsetout = annealer.get_q();
        
        TEST_ASSERT(compareSolutions(bsetin, bsetout));
    }

}
