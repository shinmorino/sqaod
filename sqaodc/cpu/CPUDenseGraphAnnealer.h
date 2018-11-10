/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>

namespace sqaod_cpu {

namespace sq = sqaod;

template<class real>
class CPUDenseGraphAnnealer : public sq::DenseGraphAnnealer<real> {

    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;

public:
    CPUDenseGraphAnnealer();
    ~CPUDenseGraphAnnealer();

    void seed(unsigned long long seed);

    sq::Algorithm selectAlgorithm(enum sq::Algorithm algo);
    
    /* void getProblemSize(SizeType *N) const; */

    void setQUBO(const Matrix &W, sq::OptimizeMethod om = sq::optMinimize);

    void setHamiltonian(const Vector &h, const Matrix &J, real c = real(0.));

    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;

    const Vector &get_E() const;

    const sq::BitSetArray &get_x() const;

    void set_q(const sq::BitSet &x);

    void set_qset(const sq::BitSetArray &x);

    const sq::BitSetArray &get_q() const;

    void getHamiltonian(Vector *h, Matrix *J, real *c) const;

    void randomizeSpin();

    void prepare();

    void calculate_E();

    void makeSolution();

    real getSystemE(real G, real beta) const;

    void annealOneStep(real G, real beta) {
        (this->*annealMethod_)(G, beta);
    }

private:    
    typedef void (CPUDenseGraphAnnealer<real>::*AnnealMethod)(real G, real beta);
    AnnealMethod annealMethod_;
    
    void annealOneStepNaive(real G, real beta);
    void annealOneStepColoring(real G, real beta);
    void annealOneStepColoringParallel(real G, real beta);
    /* actual annealing function for annealOneStepColored. */
    void annealColoredPlane(real G, real beta);
    void annealColoredPlaneParallel(real G, real beta);
    /* simulated annealing */
    void annealOneStepSANaive(real kT, real _);

    void syncBits();
    
    sq::Random *random_;
    int nWorkers_;
    mutable Vector E_;
    sq::BitSetArray bitsX_;
    sq::BitSetArray bitsQ_;
    Matrix matQ_;
    Vector h_;
    Matrix J_;
    real c_;
    
    typedef CPUDenseGraphAnnealer<real> This;
    typedef sq::DenseGraphAnnealer<real> Base;
    using Base::selectDefaultAlgorithm;
    using Base::selectDefaultSAAlgorithm;
    using Base::om_;
    using Base::N_;
    using Base::m_;
    using Base::algo_;
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
