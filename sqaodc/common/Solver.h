#pragma once

#include <sqaodc/common/Matrix.h>
#include <sqaodc/common/Preference.h>

namespace sqaod {


enum OptimizeMethod {
    /* These values should sync with python bind. */
    optNone = -1,
    optMinimize = 0,
    optMaximize = 1,
};


template<class real>
struct Solver : NullBase {
    virtual ~Solver() WAR_VC_NOTHROW { }

    virtual Algorithm selectAlgorithm(Algorithm algo) = 0;
    
    virtual Algorithm getAlgorithm() const = 0;

    virtual Preferences getPreferences() const = 0;

    virtual void setPreference(const Preference &pref) = 0;

    template<class V>
    void setPreference(enum PreferenceName name, const V value) {
        sqaod::Preference pref(name, value);
        setPreference(pref);
    }

    void setPreferences(const Preferences &prefs);

    virtual const VectorType<real> &get_E() const = 0;

    virtual void prepare() = 0;

    virtual void calculate_E() = 0;

    virtual void makeSolution() = 0;
    
protected:
    Solver() : solverState_(solNone), om_(optNone) { }

    int solverState_;

    enum SolverState {
        solNone = 0,
        solProblemSet = 1,
        solPrepared = 2,
        solEAvailable = 4,
        solSolutionAvailable = 8,
        /* annealer-specific states */
        solRandSeedGiven = 16,
        solQSet = 32,
    };

    /* Solver state set/clear/assertions */
    void setState(SolverState state);
    void clearState(SolverState state);
    bool isRandSeedGiven() const;
    bool isProblemSet() const;
    bool isPrepared() const;
    bool isQSet() const;
    bool isEAvailable() const;
    bool isSolutionAvailable() const;
    void throwErrorIfProblemNotSet() const;
    void throwErrorIfNotPrepared() const;
    void throwErrorIfQNotSet() const;
    
    OptimizeMethod om_;
};


template<class real>
struct BFSearcher : Solver<real> {
    virtual ~BFSearcher() WAR_VC_NOTHROW { }

    virtual Algorithm selectAlgorithm(Algorithm algo);
    
    virtual Algorithm getAlgorithm() const;

    virtual void search() = 0;

protected:
    BFSearcher() { }

};


template<class real>
struct Annealer : Solver<real> {
    virtual ~Annealer() WAR_VC_NOTHROW { }

    virtual Algorithm getAlgorithm() const;

    virtual Preferences getPreferences() const;

    virtual void setPreference(const Preference &pref);

    using Solver<real>::setPreference;

    virtual void seed(unsigned long long seed) = 0;

    virtual void randomizeSpin() = 0;
    
    virtual void annealOneStep(real G, real beta) = 0;

    virtual real getSystemE(real G, real beta) const = 0;
    
protected:
    Annealer() : m_(0) { }
    sqaod::Algorithm algo_;

    void selectDefaultAlgorithm(sqaod::Algorithm algoOrg, sqaod::Algorithm algoDef, sqaod::Algorithm algoSADef);
    void selectDefaultSAAlgorithm(sqaod::Algorithm algoOrg, sqaod::Algorithm algoSADef);

    SizeType m_;
};


template<class real>
struct DenseGraphSolver {
    virtual ~DenseGraphSolver() { }

    void getProblemSize(SizeType *N) const;

    virtual void setQUBO(const MatrixType<real> &W,
                         OptimizeMethod om = sqaod::optMinimize) = 0;

    virtual const BitSetArray &get_x() const = 0;

protected:
    DenseGraphSolver() : N_(0) { }

    SizeType N_;
};



template<class real>
struct BipartiteGraphSolver {
    virtual ~BipartiteGraphSolver() { }

    void getProblemSize(SizeType *N0, SizeType *N1) const;
    
    virtual void setQUBO(const VectorType<real> &b0, const VectorType<real> &b1,
                         const MatrixType<real> &W, OptimizeMethod om = sqaod::optMinimize) = 0;

    virtual const BitSetPairArray &get_x() const = 0;

protected:
    BipartiteGraphSolver() : N0_(0), N1_(0) { }

    SizeType N0_, N1_;
};



template<class real>
struct DenseGraphBFSearcher
        : BFSearcher<real>, DenseGraphSolver<real> {
    virtual ~DenseGraphBFSearcher() WAR_VC_NOTHROW { }

    virtual Preferences getPreferences() const;

    virtual void setPreference(const Preference &pref);

    using Solver<real>::setPreference;

    virtual bool searchRange(sqaod::PackedBitSet *curXEnd) = 0;

    virtual void search();

protected:
    DenseGraphBFSearcher() :xMax_(0), tileSize_(0) { }
    
    PackedBitSet x_;
    PackedBitSet xMax_;
    SizeType tileSize_;
};


template<class real>
struct DenseGraphAnnealer
        : Annealer<real>, DenseGraphSolver<real> {
    virtual ~DenseGraphAnnealer() WAR_VC_NOTHROW { }

    virtual void setHamiltonian(const VectorType<real> &h, const MatrixType<real> &J,
                                real c = real(0.)) = 0;

    virtual void getHamiltonian(VectorType<real> *h, MatrixType<real> *J, real *c) const = 0;
    
    virtual void set_q(const BitSet &q) = 0;
    
    virtual void set_qset(const BitSetArray &q) = 0;

    virtual const BitSetArray &get_q() const = 0;

protected:
    DenseGraphAnnealer() { }
};
    

template<class real>
struct BipartiteGraphBFSearcher
        : BFSearcher<real>, BipartiteGraphSolver<real> {
    virtual ~BipartiteGraphBFSearcher() WAR_VC_NOTHROW { }

    virtual Preferences getPreferences() const;

    virtual void setPreference(const Preference &pref);

    using Solver<real>::setPreference;

    virtual bool searchRange(sqaod::PackedBitSet *curX0End, sqaod::PackedBitSet *curX1End) = 0;

    virtual void search();

protected:
    BipartiteGraphBFSearcher()
            : x0max_(0), x1max_(0), tileSize0_(0), tileSize1_(0) { }

    PackedBitSet x0_, x1_;
    PackedBitSet x0max_, x1max_;
    SizeType tileSize0_, tileSize1_;
};


template<class real>
struct BipartiteGraphAnnealer
        : Annealer<real>, BipartiteGraphSolver<real> {
    virtual ~BipartiteGraphAnnealer() WAR_VC_NOTHROW { }

    virtual void setHamiltonian(const VectorType<real> &h0, const VectorType<real> &h1,
                                const MatrixType<real> &J, real c = real(0.)) = 0;

    virtual void getHamiltonian(VectorType<real> *h0, VectorType<real> *h1,
                                MatrixType<real> *J, real *c) const = 0;

    virtual void set_q(const BitSetPair &qPair) = 0;

    virtual void set_qset(const BitSetPairArray &qPairs) = 0;

    virtual const BitSetPairArray &get_q() const = 0;

protected:
    BipartiteGraphAnnealer() { }
};


}
