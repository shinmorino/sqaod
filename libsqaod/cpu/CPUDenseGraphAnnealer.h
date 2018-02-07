/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod {

template<class real>
class CPUDenseGraphAnnealer {

    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenRowVectorType<real> EigenRowVector;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

public:
    CPUDenseGraphAnnealer();
    ~CPUDenseGraphAnnealer();

    void seed(unsigned long seed);

    void getProblemSize(SizeType *N, SizeType *m) const;

    void setProblem(const Matrix &W, OptimizeMethod om = sqaod::optMinimize);

    void setNumTrotters(SizeType m);

    const Vector &get_E() const;

    const BitsArray &get_x() const;

    void set_x(const Bits &x);

    const BitsArray &get_q() const;

    void get_hJc(Vector *h, Matrix *J, real *c) const;

    void randomize_q();

    void calculate_E();

    void initAnneal();

    void finAnneal();

    void annealOneStep(real G, real kT);
    
private:    
    void syncBits();
    int annState_;
    
    CPURandom random_;
    SizeType N_, m_;
    OptimizeMethod om_;
    Vector E_;
    BitsArray bitsX_;
    BitsArray bitsQ_;
    EigenMatrix matQ_;
    EigenRowVector h_;
    EigenMatrix J_;
    real c_;
};

}
