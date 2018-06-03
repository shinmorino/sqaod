/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
#include <sqaodc/common/internal/RangeMap.h>
#endif

namespace sqaod_cpu {

namespace sq = sqaod;

/* forwarded decl. */
template<class real> struct CPUDenseGraphBatchSearch;

template<class real>
class CPUDenseGraphBFSearcher : public sq::DenseGraphBFSearcher<real> {
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;

    typedef sqaod_cpu::CPUDenseGraphBatchSearch<real> BatchSearcher;
    
public:
    CPUDenseGraphBFSearcher();
    ~CPUDenseGraphBFSearcher();

    /* void getProblemSize(SizeType *N) const; */

    void setQUBO(const Matrix &W, sq::OptimizeMethod om = sq::optMinimize);

    sq::Preferences getPreferences() const;

    /* void setPreference(const Preference &pref); */

    const Vector &get_E() const;

    const sq::BitSetArray &get_x() const;
    
    void prepare();

    void calculate_E();
    
    void makeSolution();

    bool searchRange(sq::PackedBitSet *curXEnd) {
        return (this->*searchMethod_)(curXEnd);
    }

    /* void search(); */
    
private:    
    bool searchRangeSingleThread(sq::PackedBitSet *curXEnd);
    bool searchRangeParallel(sq::PackedBitSet *curXEnd);

    typedef bool (CPUDenseGraphBFSearcher::*SearchMethod)(sq::PackedBitSet *);
    SearchMethod searchMethod_;

    Matrix W_;
    real Emin_;
    Vector E_;
    sq::BitSetArray xList_;

    int nMaxThreads_;
    BatchSearcher *searchers_;

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
    sqaod_internal::RangeMap rangeMap_;
#endif
    
    typedef CPUDenseGraphBFSearcher<real> This;
    typedef sq::DenseGraphBFSearcher<real> Base;
    using Base::N_;
    using Base::om_;
    using Base::tileSize_;
    using Base::x_;
    using Base::xMax_;
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
