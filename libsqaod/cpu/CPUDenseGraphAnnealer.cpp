#include "CPUDenseGraphAnnealer.h"
#include "CPUFormulas.h"
#include <common/Common.h>

namespace sqd = sqaod;


template<class real>
sqd::CPUDenseGraphAnnealer<real>::CPUDenseGraphAnnealer() {
    m_ = -1;
    annState_ = annNone;
    annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepNaive;
}

template<class real>
sqd::CPUDenseGraphAnnealer<real>::~CPUDenseGraphAnnealer() {
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::seed(unsigned long seed) {
    random_.seed(seed);
    annState_ |= annRandSeedGiven;
}


template<class real>
void sqd::CPUDenseGraphAnnealer<real>::selectAlgorithm(enum Algorithm algo) {
    switch (algo) {
    case algoNaive:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepNaive;
        break;
    case algoColored:
    case algoDefault:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColored;
        break;
    default:
        log("Uknown algo, %s, defaulting to %s.", algoToName(algo), algoToName(algoColored));
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColored;
    }
}

template<class real>
enum sqd::Algorithm sqd::CPUDenseGraphAnnealer<real>::algorithm() const {
    if (annealMethod_ == &CPUDenseGraphAnnealer::annealOneStepNaive)
        return algoNaive;
    if (annealMethod_ == &CPUDenseGraphAnnealer::annealOneStepColored)
        return algoColored;
    abort_("Must not reach here.");
    return algoDefault; /* to suppress warning. */
}


template<class real>
void sqd::CPUDenseGraphAnnealer<real>::getProblemSize(SizeType *N, SizeType *m) const {
    *N = N_;
    *m = m_;
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::setProblem(const Matrix &W, OptimizeMethod om) {
    throwErrorIf(!isSymmetric(W), "W is not symmetric.");
    N_ = W.rows;
    h_.resize(1, N_);
    J_.resize(N_, N_);

    Vector h(mapFrom(h_));
    Matrix J(mapFrom(J_));
    DGFuncs<real>::calculate_hJc(&h, &J, &c_, W);
    om_ = om;
    if (om_ == sqd::optMaximize) {
        h_ *= real(-1.);
        J_ *= real(-1.);
        c_ *= real(-1.);
    }
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::setNumTrotters(SizeType m) {
    throwErrorIf(m <= 0, "# trotters must be a positive integer.");
    m_ = m;
    bitsX_.reserve(m_);
    bitsQ_.reserve(m_);
    matQ_.resize(m_, N_);;
    E_.resize(m_);
    annState_ |= annNTrottersGiven;
}

template<class real>
const sqd::VectorType<real> &sqd::CPUDenseGraphAnnealer<real>::get_E() const {
    return E_;
}

template<class real>
const sqd::BitsArray &sqd::CPUDenseGraphAnnealer<real>::get_x() const {
    return bitsX_;
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::set_x(const Bits &x) {
    EigenRowVector ex = mapToRowVector(cast<real>(x));
    matQ_.rowwise() = (ex.array() * 2 - 1).matrix();
    annState_ |= annQSet;
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::get_hJc(Vector *h, Matrix *J, real *c) const {
    mapToRowVector(*h) = h_;
    mapTo(*J) = J_;
    *c = c_;
}

template<class real>
const sqd::BitsArray &sqd::CPUDenseGraphAnnealer<real>::get_q() const {
    return bitsQ_;
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::randomize_q() {
    real *q = matQ_.data();
    for (int idx = 0; idx < IdxType(N_ * m_); ++idx)
        q[idx] = random_.randInt(2) ? real(1.) : real(-1.);
    annState_ |= annQSet;
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::initAnneal() {
    if (!(annState_ & annRandSeedGiven))
        random_.seed();
    annState_ |= annRandSeedGiven;
    if (!(annState_ & annNTrottersGiven))
        setNumTrotters((N_) / 4);
    annState_ |= annNTrottersGiven;
    if (!(annState_ & annQSet))
        randomize_q();
    annState_ |= annQSet;
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::finAnneal() {
    syncBits();
    calculate_E();
}


template<class real>
void sqd::CPUDenseGraphAnnealer<real>::calculate_E() {
    DGFuncs<real>::calculate_E(&E_, mapFrom(h_), mapFrom(J_), c_, mapFrom(matQ_));
    if (om_ == sqd::optMaximize)
        mapToRowVector(E_) *= real(-1.);
}


template<class real>
void sqd::CPUDenseGraphAnnealer<real>::syncBits() {
    bitsX_.clear();
    bitsQ_.clear();
    for (int idx = 0; idx < IdxType(m_); ++idx) {
        Bits q = extractRow<char>(matQ_, idx);
        bitsQ_.pushBack(q);
        bitsX_.pushBack(x_from_q(q));
    }
}


template<class real>
void sqd::CPUDenseGraphAnnealer<real>::annealOneStepNaive(real G, real kT) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
        
    for (int loop = 0; loop < IdxType(N_ * m_); ++loop) {
        int x = random_.randInt(N_);
        int y = random_.randInt(m_);
        real qyx = matQ_(y, x);
        real sum = J_.row(x).dot(matQ_.row(y));
        real dE = - twoDivM * qyx * (h_(x) + sum);
        int neibour0 = (y == 0) ? m_ - 1 : y - 1;
        int neibour1 = (y == m_ - 1) ? 0 : y + 1;
        dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
        real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE / kT);
        if (threshold > random_.random<real>())
            matQ_(y, x) = - qyx;
    }
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::annealColoredPlane(real G, real kT, int stepOffset) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
        
    for (int y = 0; y < IdxType(m_); ++y) {
        int offset = (stepOffset + y) % 2;
        int x = (offset + 2 * random_.randInt32()) % N_;
        real qyx = matQ_(y, x);
        real sum = J_.row(x).dot(matQ_.row(y));
        real dE = - twoDivM * qyx * (h_(x) + sum);
        int neibour0 = (y == 0) ? m_ - 1 : y - 1;
        int neibour1 = (y == m_ - 1) ? 0 : y + 1;
        dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
        real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE / kT);
        if (threshold > random_.random<real>())
            matQ_(y, x) = - qyx;
    }
}

template<class real>
void sqd::CPUDenseGraphAnnealer<real>::annealOneStepColored(real G, real kT) {
    int stepOffset = random_.randInt(2);
    for (int idx = 0; idx < (IdxType)N_; ++idx)
        annealColoredPlane(G, kT, (stepOffset + idx) & 1);
}


template class sqd::CPUDenseGraphAnnealer<float>;
template class sqd::CPUDenseGraphAnnealer<double>;
