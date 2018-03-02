/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>

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

    void setProblem(const Vector &b0, const Vector &b1, const Matrix &W,
                    sq::OptimizeMethod om = sq::optMinimize);

    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;
    
    const sq::BitsPairArray &get_x() const;

    const Vector &get_E() const;

    void initSearch();

    void finSearch();

    void searchRange(sq::PackedBits iBegin0, sq::PackedBits iEnd0,
                     sq::PackedBits iBegin1, sq::PackedBits iEnd1);

    /* void search(); */
    
private:    
    Vector b0_, b1_;
    Matrix W_;
    real Emin_;
    Vector E_;
    sq::BitsPairArray xPairList_;

    int nProcs_;
    BatchSearcher *searchers_;
    
    typedef sq::BipartiteGraphBFSearcher<real> Base;
    using Base::om_;
    using Base::N0_;
    using Base::N1_;
    using Base::tileSize0_;
    using Base::tileSize1_;
    using Base::x0max_;
    using Base::x1max_;

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
