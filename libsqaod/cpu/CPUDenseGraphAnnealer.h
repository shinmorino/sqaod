/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <common/EigenBridge.h>

namespace sqaod {

template<class real>
class CPUDenseGraphAnnealer : public DenseGraphAnnealer<real> {

    typedef EigenMatrixType<real> EigenMatrix;
    typedef EigenRowVectorType<real> EigenRowVector;
    typedef MatrixType<real> Matrix;
    typedef VectorType<real> Vector;

public:
    CPUDenseGraphAnnealer();
    ~CPUDenseGraphAnnealer();

    void seed(unsigned int seed);

    Algorithm selectAlgorithm(enum Algorithm algo);

    Algorithm getAlgorithm() const;
    
    /* void getProblemSize(SizeType *N) const; */

    void setProblem(const Matrix &W, OptimizeMethod om = sqaod::optMinimize);

    /* void setPreference(const Preference &pref); */

    /* Preferences getPreferences() const; */

    const Vector &get_E() const;

    const BitsArray &get_x() const;

    void set_x(const Bits &x);

    const BitsArray &get_q() const;

    void get_hJc(Vector *h, Matrix *J, real *c) const;

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
    typedef void (CPUDenseGraphAnnealer<real>::*AnnealMethod)(real G, real kT);
    AnnealMethod annealMethod_;

    /* actual annealing function for annealOneStepColored. */
    void annealColoredPlane(real G, real kT, int offset);

    void syncBits();
    
    Random *random_;
    int nProcs_;
    Vector E_;
    BitsArray bitsX_;
    BitsArray bitsQ_;
    EigenMatrix matQ_;
    EigenRowVector h_;
    EigenMatrix J_;
    real c_;

    typedef DenseGraphAnnealer<real> Base;
    using Base::om_;
    using Base::N_;
    using Base::m_;
    /* annealer state */
    using Base::solRandSeedGiven;
    using Base::solInitialized;
    using Base::solProblemSet;
    using Base::solQSet;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::clearState;
    using Base::isRandSeedGiven;
    using Base::isProblemSet;
    using Base::isInitialized;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotInitialized;
    using Base::throwErrorIfQNotSet;
    using Base::throwErrorIfSolutionNotAvailable;
};

}
