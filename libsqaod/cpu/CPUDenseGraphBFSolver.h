/* -*- c++ -*- */
#ifndef CPU_DENSEGRAPHBRUTEFORCESOLVER_H__
#define CPU_DENSEGRAPHANNEALER_H__

#include <cpu/Random.h>
#include <cpu/Traits.h>
#include <Eigen/Core>

namespace sqaod {

template<class real>
class CPUDenseGraphBFSolver {
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<real, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> ColumnVector;

public:
    CPUDenseGraphBFSolver();
    ~CPUDenseGraphBFSolver();

    void seed(unsigned long seed);

    void setProblemSize(int N);

    void getProblemSize(int *N) const;

    void setProblem(const real *W, OptimizeMethod om);

    const BitMatrix &get_x() const;

    real get_E() const;

    void initSearch();

    void searchRange(unsigned long long iBegin, unsigned long long iEnd);

    void search();
    
private:    
    Random random_;
    int N_;
    OptimizeMethod om_;
    unsigned long long xMax_;
    real E_;
    PackedBitsArray xList_;
    mutable BitMatrix bitX_;
    Matrix matX_;
    Matrix W_;
};

}

#endif

