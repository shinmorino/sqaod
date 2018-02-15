#include "CPUBipartiteGraphAnnealer.h"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>
#include "CPUFormulas.h"


using namespace sqaod;

template<class real>
CPUBipartiteGraphAnnealer<real>::CPUBipartiteGraphAnnealer() {
    m_ = -1;
    annState_ = annNone;
    annealMethod_ = &CPUBipartiteGraphAnnealer::annealOneStepColoring;
#ifdef _OPENMP
    nProcs_ = omp_get_num_procs();
    log("# processors: %d", nProcs_);
#else
    nProcs_ = 1;
#endif
    random_ = new Random[nProcs_];
}

template<class real>
CPUBipartiteGraphAnnealer<real>::~CPUBipartiteGraphAnnealer() {
    delete [] random_;
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::seed(unsigned long seed) {
    for (int idx = 0; idx < nProcs_; ++idx)
        random_[idx].seed(seed + 17 * idx);
    annState_ |= annRandSeedGiven;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::selectAlgorithm(Algorithm algo) {
    switch (algo) {
    case algoNaive:
        annealMethod_ = &CPUBipartiteGraphAnnealer::annealOneStepNaive;
        break;
    case algoColoring:
    case algoDefault:
        annealMethod_ = &CPUBipartiteGraphAnnealer::annealOneStepColoring;
        break;
    default:
        log("Uknown algo, %s, defaulting to %s.", algoToName(algo), algoToName(algoColoring));
        annealMethod_ = &CPUBipartiteGraphAnnealer::annealOneStepColoring;
    }
}

template<class real>
Algorithm CPUBipartiteGraphAnnealer<real>::algorithm() const {
    if (annealMethod_ == &CPUBipartiteGraphAnnealer::annealOneStepNaive)
        return algoNaive;
    if (annealMethod_ == &CPUBipartiteGraphAnnealer::annealOneStepColoring)
        return algoColoring;
    abort_("Must not reach here.");
    return algoDefault; /* to suppress warning. */
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::getProblemSize(SizeType *N0, SizeType *N1) const {
    *N0 = N0_;
    *N1 = N1_;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::setProblem(const Vector &b0, const Vector &b1,
                                                 const Matrix &W, OptimizeMethod om) {
    N0_ = (int)b0.size;
    N1_ = (int)b1.size;
    h0_.resize(N0_);
    h1_.resize(N1_);
    J_.resize(N1_, N0_);
    Vector h0(mapFrom(h0_)), h1(mapFrom(h1_));
    Matrix J(mapFrom(J_));
    BGFuncs<real>::calculate_hJc(&h0, &h1, &J, &c_, b0, b1, W);
    
    om_ = om;
    if (om_ == optMaximize) {
        h0_ *= real(-1.);
        h1_ *= real(-1.);
        J_ *= real(-1.);
        c_ *= real(-1.);
    }
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::setNumTrotters(SizeType m) {
    throwErrorIf(m <= 0, "# trotters must be a positive integer.");
    m_ = m;
    matQ0_.resize(m_, N0_);
    matQ1_.resize(m_, N1_);
    E_.resize(m_);
    annState_ |= annNTrottersGiven;
}

template<class real>
const BitsPairArray &CPUBipartiteGraphAnnealer<real>::get_x() const {
    return bitsPairX_;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::set_x(const Bits &x0, const Bits &x1) {
    EigenRowVector ex0 = mapToRowVector(x0).cast<real>();
    EigenRowVector ex1 = mapToRowVector(x1).cast<real>();
    matQ0_.rowwise() = (ex0.array() * 2 - 1).matrix();
    matQ1_.rowwise() = (ex1.array() * 2 - 1).matrix();
    annState_ |= annQSet;
}


template<class real>
const VectorType<real> &CPUBipartiteGraphAnnealer<real>::get_E() const {
    return E_;
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::get_hJc(Vector *h0, Vector *h1,
                                              Matrix *J, real *c) const {
    mapToRowVector(*h0) = h0_;
    mapToRowVector(*h1) = h1_;
    mapTo(*J) = J_;
    *c = c_;
}


template<class real>
const BitsPairArray &CPUBipartiteGraphAnnealer<real>::get_q() const {
    return bitsPairQ_;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::randomize_q() {
    real *q = matQ0_.data();
#ifndef _OPENMP
    Random &random = random_[0];
#else
#pragma omp parallel
    Random &random = random_[omp_get_thread_num()];
#pragma omp for
#endif
    for (int idx = 0; idx < IdxType(N0_ * m_); ++idx) {
        q[idx] = random.randInt(2) ? real(1.) : real(-1.);
    }
    q = matQ1_.data();
#ifdef _OPENMP
#pragma omp for
#endif
    for (int idx = 0; idx < IdxType(N1_ * m_); ++idx) {
        q[idx] = random.randInt(2) ? real(1.) : real(-1.);
    }
    annState_ |= annQSet;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::calculate_E() {
    BGFuncs<real>::calculate_E(&E_, mapFrom(h0_), mapFrom(h1_), mapFrom(J_), c_,
                               mapFrom(matQ0_), mapFrom(matQ1_));
    if (om_ == optMaximize)
        mapToRowVector(E_) *= real(-1.);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::initAnneal() {
    if (!(annState_ & annRandSeedGiven))
        for (int idx = 0; idx < nProcs_; ++idx)
            random_[idx].seed();
    annState_ |= annRandSeedGiven;
    if (!(annState_ & annNTrottersGiven))
        setNumTrotters((N0_ + N1_) / 4);
    annState_ |= annNTrottersGiven;
    if (!(annState_ & annQSet))
        randomize_q();
    annState_ |= annQSet;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::finAnneal() {
    syncBits();
    calculate_E();
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepNaive(real G, real kT) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
    Random &random = random_[0];
    int N = N0_ + N1_;
    for (int loop = 0; loop < IdxType(N * m_); ++loop) {
        int x = random.randInt(N);
        int y = random.randInt(m_);
        if (x < N0_) {
            real qyx = matQ0_(y, x);
            real sum = J_.transpose().row(x).dot(matQ1_.row(y));
            real dE = - twoDivM * qyx * (h0_(x) + sum);
            int neibour0 = (y == 0) ? m_ - 1 : y - 1;
            int neibour1 = (y == m_ - 1) ? 0 : y + 1;
            dE -= qyx * (matQ0_(neibour0, x) + matQ0_(neibour1, x)) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE / kT);
            if (threshold > random.random<real>())
                matQ0_(y, x) = - qyx;
        }
        else {
            x -= N0_;
            real qyx = matQ1_(y, x);
            real sum = J_.row(x).dot(matQ0_.row(y));
            real dE = - twoDivM * qyx * (h1_(x) + sum);
            int neibour0 = (y == 0) ? m_ - 1 : y - 1;
            int neibour1 = (y == m_ - 1) ? 0 : y + 1;
            dE -= qyx * (matQ1_(neibour0, x) + matQ1_(neibour1, x)) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE / kT);
            if (threshold > random.random<real>())
                matQ1_(y, x) = - qyx;
        }
    }
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepColoring(real G, real kT) {
    annealHalfStep(N1_, matQ1_, h1_, J_, matQ0_, G, kT);
    annealHalfStep(N0_, matQ0_, h0_, J_.transpose(), matQ1_, G, kT);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::
annealHalfStep(int N, EigenMatrix &qAnneal,
               const EigenRowVector &h, const EigenMatrix &J,
               const EigenMatrix &qFixed, real G, real kT) {
    EigenMatrix dEmat = J * qFixed.transpose();
    real twoDivM = real(2.) / m_;
    real tempCoef = std::log(std::tanh(G / kT / m_)) / kT;
    real invKT = real(1.) / kT;

#ifndef _OPENMP
    {
        Random &random = random_[0];
#else
#pragma omp parallel
    {
        Random &random = random_[omp_get_thread_num()];
#pragma omp for
#endif
        for (int im = 0; im < m_; im += 2) {
            for (int iq = 0; iq < N; ++iq) {
                real q = qAnneal(im, iq);
                real dE = - twoDivM * q * (h[iq] + dEmat(iq, im));
                int mNeibour0 = (im + m_ - 1) % m_;
                int mNeibour1 = (im + 1) % m_;
                dE -= q * (qAnneal(mNeibour0, iq) + qAnneal(mNeibour1, iq)) * tempCoef;
                real thresh = dE < real(0.) ? real(1.) : std::exp(- dE * invKT);
                if (thresh > random.random<real>())
                    qAnneal(im, iq) = -q;
            }
        }
#ifdef _OPENMP
#pragma omp for
#endif
        for (int im = 1; im < m_; im += 2) {
            for (int iq = 0; iq < N; ++iq) {
                real q = qAnneal(im, iq);
                real dE = - twoDivM * q * (h[iq] + dEmat(iq, im));
                int mNeibour0 = (im + m_ - 1) % m_;
                int mNeibour1 = (im + 1) % m_;
                dE -= q * (qAnneal(mNeibour0, iq) + qAnneal(mNeibour1, iq)) * tempCoef;
                real thresh = dE < real(0.) ? real(1.) : std::exp(-dE * invKT);
                if (thresh > random.random<real>())
                    qAnneal(im, iq) = -q;
            }
        }
    }
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::syncBits() {
    bitsPairX_.clear();
    bitsPairQ_.clear();
    Bits x0, x1;
    for (int idx = 0; idx < IdxType(m_); ++idx) {
        Bits q0 = extractRow<char>(matQ0_, idx);
        Bits q1 = extractRow<char>(matQ1_, idx);
        bitsPairQ_.pushBack(BitsPairArray::ValueType(q0, q1));
        Bits x0 = x_from_q(q0), x1 = x_from_q(q1);
        bitsPairX_.pushBack(BitsPairArray::ValueType(x0, x1));
    }
}


template class sqaod::CPUBipartiteGraphAnnealer<float>;
template class sqaod::CPUBipartiteGraphAnnealer<double>;
