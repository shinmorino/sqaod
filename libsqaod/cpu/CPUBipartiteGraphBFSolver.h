/* -*- c++ -*- */
#ifndef CPU_BIPARTITEGRAPH_BF_SOLVER_H__
#define CPU_BIPARTITEGRAPH_BF_SOLVER_H__

#include <cpu/Random.h>
#include <cpu/Traits.h>
#include <Eigen/Core>

namespace sqaod {

template<class real>
class CPUBipartiteGraphBFSolver {
    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<real, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> ColumnVector;

public:
    CPUBipartiteGraphBFSolver();
    ~CPUBipartiteGraphBFSolver();

    void seed(unsigned long seed);

    void getProblemSize(int *N0, int *N1) const;

    void setProblem(const real *b0, const real *b1, const real *W,
                    int N0, int N1, OptimizeMethod om);

    void setTileSize(int tileSize0, int tileSize1);

    const BitsPairArray &get_x() const;

    real get_E() const;

    void initSearch();

    void searchRange(PackedBits iBegin0, PackedBits iEnd0,
                     PackedBits iBegin1, PackedBits iEnd1);

    void search();
    
private:    
    Random random_;
    int N0_, N1_;
    RowVector b0_, b1_;
    Matrix W_;
    OptimizeMethod om_;
    int tileSize0_, tileSize1_;
    unsigned long long x0max_, x1max_;
    real E_;
    PackedBitsPairArray xPackedParis_;
    mutable BitsPairArray xPairs_;
};

}

#endif
