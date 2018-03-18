#include "BFSearcherRangeCoverageTest.h"
#include "utils.h"

BFSearcherRangeCoverageTest::BFSearcherRangeCoverageTest(void)
        : MinimalTestSuite("BFSearcherRangeCoverageTest") {
}


BFSearcherRangeCoverageTest::~BFSearcherRangeCoverageTest(void) {
}


void BFSearcherRangeCoverageTest::setUp() {
    
}

void BFSearcherRangeCoverageTest::tearDown() {
    
}
    
void BFSearcherRangeCoverageTest::run(std::ostream &ostm) {
#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
    testcase("DenseGraphBFSearcher") {
        sqaod::cpu::DenseGraphBFSearcher<float> searcher;
        sq::MatrixType<float> W = createRandomSymmetricMatrix<float>(12);
        searcher.setPreference(sq::Preference(sq::pnTileSize, 61));
        searcher.setQUBO(W);
        searcher.prepare();
        searcher.search();
        searcher.makeSolution();
        TEST_ASSERT(true);
    }
    testcase("BipartiteGraphBFSearcher") {
        sq::SizeType N0 = 12;
        sq::SizeType N1 = 10;
        sqaod::cpu::BipartiteGraphBFSearcher<float> searcher;
        sqaod::VectorType<float> b0 = testVec<float>(N0);
        sqaod::VectorType<float> b1 = testVec<float>(N1);
        sq::MatrixType<float> W = testMat<float>(sq::Dim(N1, N0));

        searcher.setPreference(sq::Preference(sq::pnTileSize0, 61));
        searcher.setPreference(sq::Preference(sq::pnTileSize1, 37));
        searcher.setQUBO(b0, b1, W);
        searcher.prepare();
        searcher.search();
        searcher.makeSolution();
        TEST_ASSERT(true);
    }
#else
    testcase("Not enabled") {
        std::cerr << "define SQAODC_ENABLE_RANGE_COVERAGE_TEST to run this test." << std::endl;
        TEST_FAIL;
    }
#endif
}
