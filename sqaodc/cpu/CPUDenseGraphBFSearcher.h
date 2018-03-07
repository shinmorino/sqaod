/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>


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

    void setProblem(const Matrix &W, sq::OptimizeMethod om = sq::optMinimize);

    sq::Preferences getPreferences() const;

    /* void setPreference(const Preference &pref); */

    const Vector &get_E() const;

    const sq::BitsArray &get_x() const;
    
    void initSearch();

    void finSearch();

    void searchRange(unsigned long long iBegin, unsigned long long iEnd);

    /* void search(); */
    
private:    
    Matrix W_;
    real Emin_;
    Vector E_;
    sq::BitsArray xList_;

    int nMaxThreads_;
    BatchSearcher *searchers_;

    typedef sq::DenseGraphBFSearcher<real> Base;
    using Base::N_;
    using Base::om_;
    using Base::tileSize_;
    using Base::xMax_;
    /* searcher state */
    using Base::solInitialized;
    using Base::solProblemSet;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::clearState;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotInitialized;
    using Base::throwErrorIfSolutionNotAvailable;
};

}
