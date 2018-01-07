/* -*- c++ -*- */
#ifndef CUDADENSEGRAPHANNEALER_H__
#define CUDADENSEGRAPHANNEALER_H__

#include <common/Common.h>
#include <cuda/DeviceRandom.h>

namespace sqaod {


template<class real>
class CUDADenseGraphAnnealer {
public:
    CUDADenseGraphAnnealer();
    ~CUDADenseGraphAnnealer();

    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

    void seed(unsigned long seed);

    void setProblem(const Matrix &W, OptimizeMethod om);

    void getProblemSize(int *N, int *m) const;

    void setNumTrotters(int m);

    const Vector &get_E() const;

    const BitsArray &get_x() const;

    const BitsArray &get_q() const;

    void get_hJc(Vector *h, Matrix *J, real *c) const;

    void randomize_q();

    void calculate_E();

    void initAnneal();

    void finAnneal();

    void annealOneStep(real G, real kT);
    
private:
    void allocate(int N, int m);
    void deallocate();
    void syncBits();

    int N_, m_;
};


}

#endif
