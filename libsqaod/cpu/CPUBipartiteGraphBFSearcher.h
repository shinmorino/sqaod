/* -*- c++ -*- */
#ifndef CPU_BIPARTITEGRAPH_BF_SOLVER_H__
#define CPU_BIPARTITEGRAPH_BF_SOLVER_H__

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod {

template<class real>
class CPUBipartiteGraphBFSearcher : public BipartiteGraphBFSearcher<real> {
    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenRowVectorType<real> EigenRowVector;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

    typedef BipartiteGraphBFSearcher<real> Base;
    using Base::om_;
    using Base::N0_;
    using Base::N1_;
    using Base::tileSize0_;
    using Base::tileSize1_;
    using Base::x0max_;
    using Base::x1max_;

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
    EigenRowVector b0_, b1_;
    EigenMatrix W_;
    real minE_;
    Vector E_;
    PackedBitsPairArray xPackedPairs_;
    BitsPairArray xPairs_;
};

}

#endif
