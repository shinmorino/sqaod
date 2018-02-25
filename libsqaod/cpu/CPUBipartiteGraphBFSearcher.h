/* -*- c++ -*- */
#ifndef CPU_BIPARTITEGRAPH_BF_SOLVER_H__
#define CPU_BIPARTITEGRAPH_BF_SOLVER_H__

#include <common/Common.h>


namespace sqaod_cpu {
/* forwarded decl. */
template<class real> struct CPUBipartiteGraphBatchSearch;
}


namespace sqaod {

template<class real>
class CPUBipartiteGraphBFSearcher : public BipartiteGraphBFSearcher<real> {
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

    typedef sqaod_cpu::CPUBipartiteGraphBatchSearch<real> BatchSearcher;
    
public:
    CPUBipartiteGraphBFSearcher();
    ~CPUBipartiteGraphBFSearcher();

    /* void getProblemSize(int *N0, int *N1) const; */

    void setProblem(const Vector &b0, const Vector &b1, const Matrix &W,
                    OptimizeMethod om = sqaod::optMinimize);

    /* void setPreference(const Preference &pref); */

    /* Preferences getPreferences() const; */
    
    const BitsPairArray &get_x() const;

    const Vector &get_E() const;

    void initSearch();

    void finSearch();

    void searchRange(PackedBits iBegin0, PackedBits iEnd0,
                     PackedBits iBegin1, PackedBits iEnd1);

    /* void search(); */
    
private:    
    Vector b0_, b1_;
    Matrix W_;
    real Emin_;
    Vector E_;
    BitsPairArray xPairList_;

    int nProcs_;
    BatchSearcher *searchers_;
    
    typedef BipartiteGraphBFSearcher<real> Base;
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

#endif
