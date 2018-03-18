/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
#include <sqaodc/common/RangeMap.h>
#endif

namespace sqaod_cpu {

namespace sq = sqaod;

/* forwarded decl. */
template<class real> struct CPUBipartiteGraphBatchSearch;

template<class real>
class CPUBipartiteGraphBFSearcher : public sq::BipartiteGraphBFSearcher<real> {
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;

    typedef sqaod_cpu::CPUBipartiteGraphBatchSearch<real> BatchSearcher;
    
public:
    CPUBipartiteGraphBFSearcher();
    ~CPUBipartiteGraphBFSearcher();

    /* void getProblemSize(int *N0, int *N1) const; */

    void setQUBO(const Vector &b0, const Vector &b1, const Matrix &W,
                 sq::OptimizeMethod om = sq::optMinimize);

    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;
    
    const sq::BitSetPairArray &get_x() const;

    const Vector &get_E() const;

    void prepare();

    void calculate_E();

    void makeSolution();

    bool searchRange(sq::PackedBitSet *currentX0, sq::PackedBitSet *currentX1);

    /* void search(); */
    
private:    
    Vector b0_, b1_;
    Matrix W_;
    real Emin_;
    Vector E_;
    sq::BitSetPairArray xPairList_;

    int nMaxThreads_;
    BatchSearcher *searchers_;

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
    sqaod_internal::RangeMapArray rangeMapArray_;
#endif
    typedef CPUBipartiteGraphBFSearcher<real> This;
    typedef sq::BipartiteGraphBFSearcher<real> Base;
    using Base::om_;
    using Base::N0_;
    using Base::N1_;
    using Base::tileSize0_;
    using Base::tileSize1_;
    using Base::x0_;
    using Base::x1_;
    using Base::x0max_;
    using Base::x1max_;
    /* searcher state */
    using Base::solPrepared;
    using Base::solProblemSet;
    using Base::solEAvailable;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::clearState;
    using Base::isEAvailable;
    using Base::isSolutionAvailable;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotPrepared;
};

}
