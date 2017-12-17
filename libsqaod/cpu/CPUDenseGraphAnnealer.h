/* -*- c++ -*- */
#ifndef CPU_DENSEGRAPHANNEALER_H__
#define CPU_DENSEGRAPHANNEALER_H__

#include <cpu/Random.h>
#include <cpu/Traits.h>
#include <Eigen/Core>

namespace sqaod {

template<class real>
class CPUDenseGraphAnnealer {

    typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<real, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Matrix<real, Eigen::Dynamic, 1> ColumnVector;

public:
    CPUDenseGraphAnnealer();
    ~CPUDenseGraphAnnealer();

    void seed(unsigned long seed);

    void getProblemSize(int *N, int *m) const;

    void setProblem(const real *W, int N, OptimizeMethod om);

    void setNumTrotters(int m);

    void randomize_q();

    const char *get_q() const;

    void get_hJc(const real **h, const real **J, real *c) const;

    const real *get_E() const;

    void calculate_E();

    void annealOneStep(real G, real kT);
    
private:    
    Random random_;
    int N_, m_;
    OptimizeMethod om_;
    ColumnVector E_;
    mutable BitMatrix bitQ_;
    Matrix matQ_;
    RowVector h_;
    Matrix J_;
    real c_;

};

}

#endif

