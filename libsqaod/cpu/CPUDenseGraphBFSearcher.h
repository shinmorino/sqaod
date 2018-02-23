/* -*- c++ -*- */
#pragma once

#include <common/Common.h>


namespace sqaod_cpu {
/* forwarded decl. */
template<class real> struct CPUDenseGraphBatchSearch;
}

namespace sqaod {

template<class real>
class CPUDenseGraphBFSearcher : public DenseGraphBFSearcher<real> {
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

    typedef DenseGraphBFSearcher<real> Base;
    using Base::N_;
    using Base::om_;
    using Base::tileSize_;
    using Base::xMax_;

    typedef sqaod_cpu::CPUDenseGraphBatchSearch<real> BatchSearcher;
    
public:
    CPUDenseGraphBFSearcher();
    ~CPUDenseGraphBFSearcher();

    /* void getProblemSize(SizeType *N) const; */

    void setProblem(const Matrix &W, OptimizeMethod om = optMinimize);

    /* Preferences getPreferences() const; */

    /* void setPreference(const Preference &pref); */

    const Vector &get_E() const;

    const BitsArray &get_x() const;
    
    void initSearch();

    void finSearch();

    void searchRange(unsigned long long iBegin, unsigned long long iEnd);

    /* void search(); */
    
private:    
    Matrix W_;
    real Emin_;
    Vector E_;
    BitsArray xList_;

    int nProcs_;
    BatchSearcher *searchers_;
};

}
