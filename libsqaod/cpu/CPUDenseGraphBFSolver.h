/* -*- c++ -*- */
#ifndef CPU_DENSEGRAPHBRUTEFORCESOLVER_H__
#define CPU_DENSEGRAPHANNEALER_H__

#include <common/Common.h>
#include <cpu/Random.h>

namespace sqaod {

template<class real>
class CPUDenseGraphBFSolver {
    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenRowVectorType<real> EigenRowVector;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

public:
    CPUDenseGraphBFSolver();
    ~CPUDenseGraphBFSolver();

    void seed(unsigned long seed);

    void getProblemSize(int *N) const;

    void setProblem(const Matrix &W, OptimizeMethod om);

    void setTileSize(int tileSize);

    const BitsArray &get_x() const;

    const Vector &get_E() const;
    
    void initSearch();

    void finSearch();

    void searchRange(unsigned long long iBegin, unsigned long long iEnd);

    void search();
    
private:    
    Random random_;
    int N_;
    OptimizeMethod om_;
    int tileSize_;
    unsigned long long xMax_;
    real minE_;
    Vector E_;
    PackedBitsArray packedXList_;
    BitsArray xList_;
    EigenMatrix matX_;
    EigenMatrix W_;
};

}

#endif

