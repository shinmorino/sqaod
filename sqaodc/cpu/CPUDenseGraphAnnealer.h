/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/common/EigenBridge.h>

namespace sqaod_cpu {

namespace sq = sqaod;

template<class real>
class CPUDenseGraphAnnealer : public sq::DenseGraphAnnealer<real> {

    typedef sq::EigenMatrixType<real> EigenMatrix;
    typedef sq::EigenRowVectorType<real> EigenRowVector;
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;

public:
    CPUDenseGraphAnnealer();
    ~CPUDenseGraphAnnealer();

    void seed(unsigned long long seed);

    sq::Algorithm selectAlgorithm(enum sq::Algorithm algo);

    sq::Algorithm getAlgorithm() const;
    
    /* void getProblemSize(SizeType *N) const; */

    void setProblem(const Matrix &W, sq::OptimizeMethod om = sq::optMinimize);

    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;

    const Vector &get_E() const;

    const sq::BitsArray &get_x() const;

    void set_x(const sq::Bits &x);

    const sq::BitsArray &get_q() const;

    void get_hJc(Vector *h, Matrix *J, real *c) const;

    void randomizeSpin();

    void prepare();

    void calculate_E();

    void makeSolution();

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
    
    sq::Random *random_;
    int nMaxThreads_;
    Vector E_;
    sq::BitsArray bitsX_;
    sq::BitsArray bitsQ_;
    EigenMatrix matQ_;
    EigenRowVector h_;
    EigenMatrix J_;
    real c_;

    typedef CPUDenseGraphAnnealer<real> This;
    typedef sq::DenseGraphAnnealer<real> Base;
    using Base::om_;
    using Base::N_;
    using Base::m_;
    /* annealer state */
    using Base::solRandSeedGiven;
    using Base::solPrepared;
    using Base::solProblemSet;
    using Base::solQSet;
    using Base::solEAvailable;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::clearState;
    using Base::isRandSeedGiven;
    using Base::isEAvailable;
    using Base::isSolutionAvailable;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotPrepared;
    using Base::throwErrorIfQNotSet;
};

}
