/* -*- c++ -*- */
#ifndef CPU_BIPARTITEGRAPH_ANNEALER_H__
#define CPU_BIPARTITEGRAPH_ANNEALER_H__

#include <common/Common.h>
#include <common/EigenBridge.h>


namespace sqaod {

template<class real>
class CPUBipartiteGraphAnnealer {
    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenRowVectorType<real> EigenRowVector;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

public:
    CPUBipartiteGraphAnnealer();
    ~CPUBipartiteGraphAnnealer();

    void seed(unsigned long seed);

    void selectAlgorithm(sqaod::Algorithm algo);

    sqaod::Algorithm algorithm() const;

    void getProblemSize(SizeType *N0, SizeType *N1) const;

    void setProblem(const Vector &b0, const Vector &b1, const Matrix &W,
                    OptimizeMethod om = optMinimize);

    void setNumTrotters(SizeType m);

    SizeType getNumTrotters() const {
        return m_;
    }

    const Vector &get_E() const;

    const BitsPairArray &get_x() const;

    void set_x(const Bits &x0, const Bits &x1);

    /* Ising machine / spins */

    void get_hJc(Vector *h0, Vector *h1, Matrix *J, real *c) const;

    const BitsPairArray &get_q() const;

    void randomize_q();

    void calculate_E();

    void initAnneal();

    void finAnneal();

    void annealOneStep(real G, real kT) {
        (this->*annealMethod_)(G, kT);
    }

    void annealOneStepNaive(real G, real kT);

    void annealOneStepColoring(real G, real kT);
    
private:
    typedef void (CPUBipartiteGraphAnnealer<real>::*AnnealMethod)(real G, real kT);
    AnnealMethod annealMethod_;
    
    void syncBits();

    void annealHalfStepColoring(int N, EigenMatrix &qAnneal,
                                const EigenRowVector &h, const EigenMatrix &J,
                                const EigenMatrix &qFixed, real G, real kT);
    int annState_;

    Random *random_;
    int nProcs_;
    SizeType N0_, N1_, m_;
    EigenRowVector h0_, h1_;
    EigenMatrix J_;
    real c_;
    OptimizeMethod om_;
    Vector E_;
    EigenMatrix matQ0_, matQ1_;
    BitsPairArray bitsPairX_;
    BitsPairArray bitsPairQ_;
};

}

#endif
