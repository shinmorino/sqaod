#include "CPUDenseGraphAnnealer.h"
#include "CPUFormulas.h"
#include <common/Common.h>

namespace sq = sqaod;

template<class real>
sq::CPUDenseGraphAnnealer<real>::CPUDenseGraphAnnealer() {
    m_ = -1;
    annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoring;
#ifdef _OPENMP
    nProcs_ = omp_get_num_procs();
    sq::log("# processors: %d", nProcs_);
#else
    nProcs_ = 1;
#endif
    random_ = new Random[nProcs_];
}

template<class real>
sq::CPUDenseGraphAnnealer<real>::~CPUDenseGraphAnnealer() {
    delete [] random_;
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::seed(unsigned int seed) {
    for (int idx = 0; idx < nProcs_; ++idx)
        random_[idx].seed(seed + 17 * idx);
    setState(solRandSeedGiven);
}


template<class real>
sq::Algorithm sq::CPUDenseGraphAnnealer<real>::selectAlgorithm(enum Algorithm algo) {
    switch (algo) {
    case algoNaive:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepNaive;
        return algoNaive;
    case algoColoring:
    case algoDefault:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoring;
        return algoColoring;
        break;
    default:
        log("Uknown algo, %s, defaulting to %s.",
            algorithmToString(algo), algorithmToString(algoColoring));
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoring;
        return algoColoring;
    }
}

template<class real>
enum sq::Algorithm sq::CPUDenseGraphAnnealer<real>::getAlgorithm() const {
    if (annealMethod_ == &CPUDenseGraphAnnealer::annealOneStepNaive)
        return algoNaive;
    if (annealMethod_ == &CPUDenseGraphAnnealer::annealOneStepColoring)
        return algoColoring;
    abort_("Must not reach here.");
    return algoDefault; /* to suppress warning. */
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::setProblem(const Matrix &W, OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");

    if (N_ != W.rows)
        clearState(solInitialized);

    N_ = W.rows;
    m_ = N_ / 4;
    h_.resize(1, N_);
    J_.resize(N_, N_);

    Vector h(mapFrom(h_));
    Matrix J(mapFrom(J_));
    DGFuncs<real>::calculate_hJc(&h, &J, &c_, W);
    om_ = om;
    if (om_ == sq::optMaximize) {
        h_ *= real(-1.);
        J_ *= real(-1.);
        c_ *= real(-1.);
    }
    setState(solProblemSet);
}



template<class real>
const sq::VectorType<real> &sq::CPUDenseGraphAnnealer<real>::get_E() const {
    throwErrorIfSolutionNotAvailable();
    return E_;
}

template<class real>
const sq::BitsArray &sq::CPUDenseGraphAnnealer<real>::get_x() const {
    throwErrorIfSolutionNotAvailable();
    return bitsX_;
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::set_x(const Bits &x) {
    throwErrorIfNotInitialized();
    EigenRowVector ex = mapToRowVector(cast<real>(x));
    matQ_ = (ex.array() * 2 - 1).matrix();
    setState(solQSet);
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::get_hJc(Vector *h, Matrix *J, real *c) const {
    throwErrorIfProblemNotSet();
    mapToRowVector(*h) = h_;
    mapTo(*J) = J_;
    *c = c_;
}

template<class real>
const sq::BitsArray &sq::CPUDenseGraphAnnealer<real>::get_q() const {
    throwErrorIfSolutionNotAvailable();
    return bitsQ_;
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::randomize_q() {
    throwErrorIfNotInitialized();
    real *q = matQ_.data();
    for (int idx = 0; idx < IdxType(N_ * m_); ++idx)
        q[idx] = random_->randInt(2) ? real(1.) : real(-1.);
    setState(solQSet);
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::initAnneal() {
    if (!isRandSeedGiven())
        random_->seed();
    setState(solRandSeedGiven);
    bitsX_.reserve(m_);
    bitsQ_.reserve(m_);
    matQ_.resize(m_, N_);;
    E_.resize(m_);

    setState(solInitialized);
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::finAnneal() {
    throwErrorIfQNotSet();
    syncBits();
    setState(solSolutionAvailable);
    calculate_E();
}


template<class real>
void sq::CPUDenseGraphAnnealer<real>::calculate_E() {
    throwErrorIfSolutionNotAvailable();
    DGFuncs<real>::calculate_E(&E_, mapFrom(h_), mapFrom(J_), c_, mapFrom(matQ_));
    if (om_ == sq::optMaximize)
        mapToRowVector(E_) *= real(-1.);
}


template<class real>
void sq::CPUDenseGraphAnnealer<real>::syncBits() {
    bitsX_.clear();
    bitsQ_.clear();
    for (int idx = 0; idx < IdxType(m_); ++idx) {
        Bits q = extractRow<char>(matQ_, idx);
        bitsQ_.pushBack(q);
        bitsX_.pushBack(x_from_q(q));
    }
}


template<class real>
void sq::CPUDenseGraphAnnealer<real>::annealOneStepNaive(real G, real kT) {
    throwErrorIfQNotSet();

    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
    Random &random = random_[0];
    for (int loop = 0; loop < IdxType(N_ * m_); ++loop) {
        int x = random.randInt(N_);
        int y = random.randInt(m_);
        real qyx = matQ_(y, x);
        real sum = J_.row(x).dot(matQ_.row(y));
        real dE = - twoDivM * qyx * (h_(x) + sum);
        int neibour0 = (y == 0) ? m_ - 1 : y - 1;
        int neibour1 = (y == m_ - 1) ? 0 : y + 1;
        dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
        real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE / kT);
        if (threshold > random.random<real>())
            matQ_(y, x) = - qyx;
    }
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::annealColoredPlane(real G, real kT, int stepOffset) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;

#ifndef _OPENMP
    {
        Random &random = random_[0];
#else
#  pragma omp parallel private(random) 
    {
        Random &random = random_[omp_get_thread_num()];
#  pragma omp for
#endif        
        for (int y = 0; y < IdxType(m_); ++y) {
            int offset = (stepOffset + y) % 2;
            int x = (offset + 2 * random.randInt32()) % N_;
            real qyx = matQ_(y, x);
            real sum = J_.row(x).dot(matQ_.row(y));
            real dE = - twoDivM * qyx * (h_(x) + sum);
            int neibour0 = (y == 0) ? m_ - 1 : y - 1;
            int neibour1 = (y == m_ - 1) ? 0 : y + 1;
            dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE / kT);
            if (threshold > random.random<real>())
                matQ_(y, x) = - qyx;
        }
    }
}

template<class real>
void sq::CPUDenseGraphAnnealer<real>::annealOneStepColoring(real G, real kT) {
    throwErrorIfQNotSet();
    
    int stepOffset = random_[0].randInt(2);
    for (int idx = 0; idx < (IdxType)N_; ++idx)
        annealColoredPlane(G, kT, (stepOffset + idx) & 1);
}


template class sq::CPUDenseGraphAnnealer<float>;
template class sq::CPUDenseGraphAnnealer<double>;
