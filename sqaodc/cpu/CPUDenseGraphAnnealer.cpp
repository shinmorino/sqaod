#include "CPUDenseGraphAnnealer.h"
#include "CPUFormulas.h"
#include <common/Common.h>
#include <time.h>

using namespace sqaod_cpu;

template<class real>
CPUDenseGraphAnnealer<real>::CPUDenseGraphAnnealer() {
    m_ = -1;
    annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoring;
#ifdef _OPENMP
    nMaxThreads_ = omp_get_max_threads();
    sq::log("# max threads: %d", nMaxThreads_);
#else
    nMaxThreads_ = 1;
#endif
    random_ = new sq::Random[nMaxThreads_];
}

template<class real>
CPUDenseGraphAnnealer<real>::~CPUDenseGraphAnnealer() {
    delete [] random_;
}

template<class real>
void CPUDenseGraphAnnealer<real>::seed(unsigned long long seed) {
    for (int idx = 0; idx < nMaxThreads_; ++idx)
        random_[idx].seed(seed + 17 * idx);
    setState(solRandSeedGiven);
}


template<class real>
sq::Algorithm CPUDenseGraphAnnealer<real>::selectAlgorithm(enum sq::Algorithm algo) {
    switch (algo) {
    case sq::algoNaive:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepNaive;
        return sq::algoNaive;
    case sq::algoColoring:
    case sq::algoDefault:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoring;
        return sq::algoColoring;
        break;
    default:
        sq::log("Uknown algo, %s, defaulting to %s.",
                sq::algorithmToString(algo), sq::algorithmToString(sq::algoColoring));
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoring;
        return sq::algoColoring;
    }
}

template<class real>
sq::Algorithm CPUDenseGraphAnnealer<real>::getAlgorithm() const {
    if (annealMethod_ == &CPUDenseGraphAnnealer::annealOneStepNaive)
        return sq::algoNaive;
    if (annealMethod_ == &CPUDenseGraphAnnealer::annealOneStepColoring)
        return sq::algoColoring;
    abort_("Must not reach here.");
    return sq::algoDefault; /* to suppress warning. */
}

template<class real>
void CPUDenseGraphAnnealer<real>::setProblem(const Matrix &W, sq::OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");

    if (N_ != W.rows)
        clearState(solInitialized);

    N_ = W.rows;
    m_ = N_ / 4;
    h_.resize(1, N_);
    J_.resize(N_, N_);

    Vector h(sq::mapFrom(h_));
    Matrix J(sq::mapFrom(J_));
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
sq::Preferences CPUDenseGraphAnnealer<real>::getPreferences() const {
    sq::Preferences prefs = Base::getPreferences();
    prefs.pushBack(sq::Preference(sq::pnDevice, "cpu"));
    return prefs;
}

template<class real>
const sq::VectorType<real> &CPUDenseGraphAnnealer<real>::get_E() const {
    throwErrorIfSolutionNotAvailable();
    return E_;
}

template<class real>
const sq::BitsArray &CPUDenseGraphAnnealer<real>::get_x() const {
    throwErrorIfSolutionNotAvailable();
    return bitsX_;
}

template<class real>
void CPUDenseGraphAnnealer<real>::set_x(const sq::Bits &x) {
    throwErrorIfNotInitialized();
    throwErrorIf(x.size != N_,
                 "Dimension of x, %d,  should be equal to N, %d.", x.size, N_);
    
    EigenRowVector ex = mapToRowVector(sq::cast<real>(x));
    matQ_ = (ex.array() * 2 - 1).matrix();
    setState(solQSet);
}

template<class real>
void CPUDenseGraphAnnealer<real>::get_hJc(Vector *h, Matrix *J, real *c) const {
    throwErrorIfProblemNotSet();
    mapToRowVector(*h) = h_;
    mapTo(*J) = J_;
    *c = c_;
}

template<class real>
const sq::BitsArray &CPUDenseGraphAnnealer<real>::get_q() const {
    throwErrorIfSolutionNotAvailable();
    return bitsQ_;
}

template<class real>
void CPUDenseGraphAnnealer<real>::randomize_q() {
    throwErrorIfNotInitialized();
    real *q = matQ_.data();
    for (int idx = 0; idx < sq::IdxType(N_ * m_); ++idx)
        q[idx] = random_->randInt(2) ? real(1.) : real(-1.);
    setState(solQSet);
}

template<class real>
void CPUDenseGraphAnnealer<real>::initAnneal() {
    if (!isRandSeedGiven())
        seed((unsigned long long)time(NULL));
    setState(solRandSeedGiven);
    bitsX_.reserve(m_);
    bitsQ_.reserve(m_);
    matQ_.resize(m_, N_);;
    E_.resize(m_);

    setState(solInitialized);
}

template<class real>
void CPUDenseGraphAnnealer<real>::finAnneal() {
    throwErrorIfQNotSet();
    syncBits();
    setState(solSolutionAvailable);
    calculate_E();
}


template<class real>
void CPUDenseGraphAnnealer<real>::calculate_E() {
    throwErrorIfSolutionNotAvailable();
    DGFuncs<real>::calculate_E(&E_, sq::mapFrom(h_), sq::mapFrom(J_), c_, sq::mapFrom(matQ_));
    if (om_ == sq::optMaximize)
        mapToRowVector(E_) *= real(-1.);
}


template<class real>
void CPUDenseGraphAnnealer<real>::syncBits() {
    bitsX_.clear();
    bitsQ_.clear();
    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        sq::Bits q = sq::extractRow<char>(matQ_, idx);
        bitsQ_.pushBack(q);
        bitsX_.pushBack(x_from_q(q));
    }
}


template<class real>
void CPUDenseGraphAnnealer<real>::annealOneStepNaive(real G, real kT) {
    throwErrorIfQNotSet();

    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
    sq::Random &random = random_[0];
    for (int loop = 0; loop < sq::IdxType(N_ * m_); ++loop) {
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
void CPUDenseGraphAnnealer<real>::annealColoredPlane(real G, real kT, int stepOffset) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
    real invKT = real(1.) / kT;
#ifndef _OPENMP
    {
        sq::Random &random = random_[0];
#else
#  pragma omp parallel
    {
        sq::Random &random = random_[omp_get_thread_num()];
#endif
        for (int yOffset = 0; yOffset < 2; ++yOffset) {
#ifndef _OPENMP
#  pragma omp for
#endif
            for (int y = yOffset; y < sq::IdxType(m_); y += 2) {
                int x = random.randInt32() % N_;
                real qyx = matQ_(y, x);
                real sum = J_.row(x).dot(matQ_.row(y));
                real dE = - twoDivM * qyx * (h_(x) + sum);
                int neibour0 = (y == 0) ? m_ - 1 : y - 1;
                int neibour1 = (y == m_ - 1) ? 0 : y + 1;
                dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
                real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * invKT);
                if (threshold > random.random<real>())
                    matQ_(y, x) = - qyx;
            }
        }
    }
}

template<class real>
void CPUDenseGraphAnnealer<real>::annealOneStepColoring(real G, real kT) {
    throwErrorIfQNotSet();
    
    int stepOffset = random_[0].randInt(2);
    for (int idx = 0; idx < (sq::IdxType)N_; ++idx)
        annealColoredPlane(G, kT, (stepOffset + idx) & 1);
}


template class CPUDenseGraphAnnealer<float>;
template class CPUDenseGraphAnnealer<double>;
