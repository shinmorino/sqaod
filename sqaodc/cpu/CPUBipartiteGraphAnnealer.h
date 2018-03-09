/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/common/EigenBridge.h>


namespace sqaod_cpu {

namespace sq = sqaod;

template<class real>
class CPUBipartiteGraphAnnealer : public sq::BipartiteGraphAnnealer<real> {
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;
    typedef sq::EigenMatrixType<real> EigenMatrix;
    typedef sq::EigenRowVectorType<real> EigenRowVector;
    
public:
    CPUBipartiteGraphAnnealer();
    ~CPUBipartiteGraphAnnealer();

    void seed(unsigned long long seed);

    sq::Algorithm selectAlgorithm(sq::Algorithm algo);

    sq::Algorithm getAlgorithm() const;

    /* void getProblemSize(SizeType *N0, SizeType *N1) const; */

    void setProblem(const Vector &b0, const Vector &b1, const Matrix &W,
                    sq::OptimizeMethod om = sq::optMinimize);

    /* FIXME: algo */
    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;

    const Vector &get_E() const;

    const sq::BitsPairArray &get_x() const;

    void set_x(const sq::Bits &x0, const sq::Bits &x1);

    /* Ising machine / spins */

    void get_hJc(Vector *h0, Vector *h1, Matrix *J, real *c) const;

    const sq::BitsPairArray &get_q() const;

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

    sq::Random *random_;
    int nMaxThreads_;
    EigenRowVector h0_, h1_;
    EigenMatrix J_;
    real c_;
    Vector E_;
    EigenMatrix matQ0_, matQ1_;
    sq::BitsPairArray bitsPairX_;
    sq::BitsPairArray bitsPairQ_;

    typedef sq::BipartiteGraphAnnealer<real> Base;
    using Base::om_;
    using Base::N0_;
    using Base::N1_;
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
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotInitialized;
    using Base::throwErrorIfQNotSet;
    using Base::throwErrorIfSolutionNotAvailable;
};

}
