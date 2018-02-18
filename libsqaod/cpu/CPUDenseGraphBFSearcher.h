/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod {

template<class real>
class CPUDenseGraphBFSearcher : public DenseGraphBFSearcher<real> {
    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenRowVectorType<real> EigenRowVector;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

    typedef DenseGraphBFSearcher<real> Base;
    using Base::N_;
    using Base::om_;
    using Base::tileSize_;
    using Base::xMax_;
    
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
    real minE_;
    Vector E_;
    PackedBitsArray packedXList_;
    BitsArray xList_;
    EigenMatrix matX_;
    EigenMatrix W_;
};

}
