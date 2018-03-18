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

    void setQUBO(const Vector &b0, const Vector &b1, const Matrix &W,
                 sq::OptimizeMethod om = sq::optMinimize);

    void setHamiltonian(const Vector &h0, const Vector &h1, const Matrix &J,
                        real c = real(0.));

    /* FIXME: algo */
    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;

    const Vector &get_E() const;

    const sq::BitSetPairArray &get_x() const;

    void set_x(const sq::BitSet &x0, const sq::BitSet &x1);

    /* Ising machine / spins */

    void getHamiltonian(Vector *h0, Vector *h1, Matrix *J, real *c) const;

    const sq::BitSetPairArray &get_q() const;

    void randomizeSpin();

    void prepare();

    void calculate_E();

    void makeSolution();

    void annealOneStep(real G, real beta) {
        (this->*annealMethod_)(G, beta);
    }

    void annealOneStepNaive(real G, real beta);

    void annealOneStepColoring(real G, real beta);
    
private:
    typedef void (CPUBipartiteGraphAnnealer<real>::*AnnealMethod)(real G, real beta);
    AnnealMethod annealMethod_;
    
    void syncBits();

    void annealHalfStepColoring(int N, EigenMatrix &qAnneal,
                                const EigenRowVector &h, const EigenMatrix &J,
                                const EigenMatrix &qFixed, real G, real beta);

    sq::Random *random_;
    int nMaxThreads_;
    EigenRowVector h0_, h1_;
    EigenMatrix J_;
    real c_;
    Vector E_;
    EigenMatrix matQ0_, matQ1_;
    sq::BitSetPairArray bitsPairX_;
    sq::BitSetPairArray bitsPairQ_;

    typedef CPUBipartiteGraphAnnealer<real> This;
    typedef sq::BipartiteGraphAnnealer<real> Base;
    using Base::om_;
    using Base::N0_;
    using Base::N1_;
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
