#include "CPUBipartiteGraphAnnealer.h"
#include <sqaodc/common/internal/ShapeChecker.h>
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>
#include "SharedFormulas.h"
#include <time.h>

namespace sqint = sqaod_internal;
using namespace sqaod_cpu;

template<class real>
CPUBipartiteGraphAnnealer<real>::CPUBipartiteGraphAnnealer() {
    m_ = -1;
    selectAlgorithm(sq::algoDefault);
#ifdef _OPENMP
    nMaxThreads_ = omp_get_max_threads();
#else
    nMaxThreads_ = 1;
#endif
    sq::log("# max threads: %d", nMaxThreads_);
    random_ = new sq::Random[nMaxThreads_];
}

template<class real>
CPUBipartiteGraphAnnealer<real>::~CPUBipartiteGraphAnnealer() {
    delete [] random_;
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::seed(unsigned long long seed) {
    for (int idx = 0; idx < nMaxThreads_; ++idx)
        random_[idx].seed(seed + 17 * idx);
    setState(solRandSeedGiven);
}

template<class real>
sq::Algorithm CPUBipartiteGraphAnnealer<real>::selectAlgorithm(sq::Algorithm algo) {
    switch (algo) {
    case sq::algoNaive:
    case sq::algoColoring:
        algo_ = algo;
        return algo_;
    case sq::algoDefault:
        algo_ = sq::algoColoring;
        return algo_;
    default:
        sq::log("Uknown algo, %s, defaulting to %s.",
            sq::algorithmToString(algo), sq::algorithmToString(sq::algoColoring));
        algo_ = sq::algoColoring;
        return algo;
    }
}

template<class real>
sq::Algorithm CPUBipartiteGraphAnnealer<real>::getAlgorithm() const {
    return algo_;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::setQUBO(const Vector &b0, const Vector &b1,
                                              const Matrix &W, sq::OptimizeMethod om) {
    sqint::quboShapeCheck(b0, b1, W, __func__);
    
    N0_ = (int)b0.size;
    N1_ = (int)b1.size;
    m_ = (N0_ + N1_) / 4; /* setting number of trotters. */
    h0_.resize(N0_);
    h1_.resize(N1_);
    J_.resize(N1_, N0_);
    Vector h0(sq::mapFrom(h0_)), h1(sq::mapFrom(h1_));
    Matrix J(sq::mapFrom(J_));
    BGFuncs<real>::calculateHamiltonian(&h0, &h1, &J, &c_, b0, b1, W);
    
    om_ = om;
    if (om_ == sq::optMaximize) {
        h0_ *= real(-1.);
        h1_ *= real(-1.);
        J_ *= real(-1.);
        c_ *= real(-1.);
    }
    setState(solProblemSet);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::
setHamiltonian(const Vector &h0, const Vector &h1, const Matrix &J, real c) {
    sqint::isingModelShapeCheck(h0, h1, J, c, __func__);
    
    N0_ = (int)h0.size;
    N1_ = (int)h1.size;
    m_ = (N0_ + N1_) / 4; /* setting number of trotters. */

    h0_ = sq::mapToRowVector(h0);
    h1_ = sq::mapToRowVector(h1);
    J_ = sq::mapTo(J);
    c_ = c;
    om_ = sq::optMinimize;
    setState(solProblemSet);
}

template<class real>
sq::Preferences CPUBipartiteGraphAnnealer<real>::getPreferences() const {
    sq::Preferences prefs = Base::getPreferences();
    prefs.pushBack(sq::Preference(sq::pnDevice, "cpu"));
    return prefs;
}

template<class real>
const sq::BitSetPairArray &CPUBipartiteGraphAnnealer<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return bitsPairX_;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::set_q(const sq::BitSetPair &qPair) {
    sqint::isingModelShapeCheck(sq::mapFrom(h0_), sq::mapFrom(h1_), sq::mapFrom(J_), c_,
                                qPair.bits0, qPair.bits1, __func__);
    throwErrorIfNotPrepared();
    throwErrorIf(qPair.bits0.size != N0_,
                 "Dimension of q0, %d,  should be equal to N0, %d.", qPair.bits0.size, N0_);
    throwErrorIf(qPair.bits1.size != N1_,
                 "Dimension of q1, %d,  should be equal to N1, %d.", qPair.bits1.size, N1_);

    EigenRowVector q0 = mapToRowVector(qPair.bits0).cast<real>();
    EigenRowVector q1 = mapToRowVector(qPair.bits1).cast<real>();
    for (int idx = 0; idx < m_; ++idx) {
        matQ0_.row(idx) = q0;
        matQ1_.row(idx) = q1;
    }

    setState(solQSet);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::set_qset(const sq::BitSetPairArray &qPairs) {
    sqint::isingModelSolutionShapeCheck(N0_, N1_, qPairs, __func__);
    m_ = qPairs.size();
    prepare();
    for (int idx = 0; idx < m_; ++idx) {
        matQ0_.row(idx) = mapToRowVector(qPairs[idx].bits0).cast<real>();
        matQ1_.row(idx) = mapToRowVector(qPairs[idx].bits1).cast<real>();
    }

    setState(solQSet);
}


template<class real>
const sq::VectorType<real> &CPUBipartiteGraphAnnealer<real>::get_E() const {
    if (!isEAvailable())
        const_cast<This*>(this)->calculate_E();
    return E_;
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::getHamiltonian(Vector *h0, Vector *h1,
                                                     Matrix *J, real *c) const {
    throwErrorIfProblemNotSet();
    *h0 = sq::mapFrom(h0_);
    *h1 = sq::mapFrom(h1_);
    *J = sq::mapFrom(J_);
    *c = c_;
}


template<class real>
const sq::BitSetPairArray &CPUBipartiteGraphAnnealer<real>::get_q() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return bitsPairQ_;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::randomizeSpin() {
    throwErrorIfNotPrepared();
#ifndef _OPENMP
    {
        sq::Random &random = random_[0];
        real *q = matQ0_.data();
#else
#pragma omp parallel
    {
        sq::Random &random = random_[omp_get_thread_num()];
        real *q = matQ0_.data();
#pragma omp for
#endif
        for (int idx = 0; idx < sq::IdxType(N0_ * m_); ++idx)
            q[idx] = random.randInt(2) ? real(1.) : real(-1.);
        q = matQ1_.data();
#ifdef _OPENMP
#pragma omp for 
#endif
        for (int idx = 0; idx < sq::IdxType(N1_ * m_); ++idx)
            q[idx] = random.randInt(2) ? real(1.) : real(-1.);
    }
    setState(solQSet);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::calculate_E() {
    throwErrorIfQNotSet();
    BGFuncs<real>::calculate_E(&E_, sq::mapFrom(h0_), sq::mapFrom(h1_), sq::mapFrom(J_), c_,
                               sq::mapFrom(matQ0_), sq::mapFrom(matQ1_));
    if (om_ == sq::optMaximize)
        mapToRowVector(E_) *= real(-1.);
    setState(solEAvailable);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::prepare() {
    if (!isRandSeedGiven())
        seed((unsigned long long)time(NULL));
    setState(solRandSeedGiven);

    matQ0_.resize(m_, N0_);
    matQ1_.resize(m_, N1_);
    E_.resize(m_);

    switch (algo_) {
    case sq::algoNaive:
        annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepNaive;
        break;
    case sq::algoColoring:
        if (nMaxThreads_ == 1)
            annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepColoring;
        else
            annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepColoringParallel;
        break;
    default:
        abort_("Must not reach here.");
    }
    
    setState(solPrepared);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::makeSolution() {
    throwErrorIfNotPrepared();
    syncBits();
    setState(solSolutionAvailable);
    calculate_E();
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepNaive(real G, real beta) {
    throwErrorIfQNotSet();
    
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G * beta / m_)) * beta;
    sq::Random &random = random_[0];
    int N = N0_ + N1_;
    for (int loop = 0; loop < sq::IdxType(N * m_); ++loop) {
        int x = random.randInt(N);
        int y = random.randInt(m_);
        if (x < N0_) {
            real qyx = matQ0_(y, x);
            real sum = J_.transpose().row(x).dot(matQ1_.row(y));
            real dE = twoDivM * qyx * (h0_(x) + sum);
            int neibour0 = (y == 0) ? m_ - 1 : y - 1;
            int neibour1 = (y == m_ - 1) ? 0 : y + 1;
            dE -= qyx * (matQ0_(neibour0, x) + matQ0_(neibour1, x)) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * beta);
            if (threshold > random.random<real>())
                matQ0_(y, x) = - qyx;
        }
        else {
            x -= N0_;
            real qyx = matQ1_(y, x);
            real sum = J_.row(x).dot(matQ0_.row(y));
            real dE = twoDivM * qyx * (h1_(x) + sum);
            int neibour0 = (y == 0) ? m_ - 1 : y - 1;
            int neibour1 = (y == m_ - 1) ? 0 : y + 1;
            dE -= qyx * (matQ1_(neibour0, x) + matQ1_(neibour1, x)) * coef;
            real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * beta);
            if (threshold > random.random<real>())
                matQ1_(y, x) = - qyx;
        }
    }
    clearState(solSolutionAvailable);
}

template<class real, class T> static inline
void tryFlip(sq::EigenMatrixType<real> &qAnneal, int im, const sq::EigenMatrixType<real> &dEmat, const sq::EigenRowVectorType<real> &h, const T &J, sq::SizeType N, sq::SizeType m, 
             real twoDivM, real beta, real coef, sq::Random &random) {
    for (int iq = 0; iq < N; ++iq) {
        real q = qAnneal(im, iq);
        real dE = twoDivM * q * (h[iq] + dEmat(im, iq));
        int mNeibour0 = (im + m - 1) % m;
        int mNeibour1 = (im + 1) % m;
        dE -= q * (qAnneal(mNeibour0, iq) + qAnneal(mNeibour1, iq)) * coef;
        real thresh = dE < real(0.) ? real(1.) : std::exp(- dE * beta);
        if (thresh > random.random<real>())
            qAnneal(im, iq) = -q;
    }
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::
annealHalfStepColoring(int N, EigenMatrix &qAnneal,
                       const EigenRowVector &h, const EigenMatrix &J,
                       const EigenMatrix &qFixed, real G, real beta) {
    real twoDivM = real(2.) / m_;
    real coef = std::log(std::tanh(G * beta / m_)) * beta;

    EigenMatrix dEmat = qFixed * J.transpose();
    sq::Random &random = random_[0];
    for (int offset = 0; offset < 2; ++offset) {
        for (int im = offset; im < m_; im += 2)
            tryFlip(qAnneal, im, dEmat, h, J, N, m_, twoDivM, beta, coef, random);
    }
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepColoring(real G, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);

    annealHalfStepColoring(N1_, matQ1_, h1_, J_, matQ0_, G, beta);
    annealHalfStepColoring(N0_, matQ0_, h0_, J_.transpose(), matQ1_, G, beta);
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::
annealHalfStepColoringParallel(int N, EigenMatrix &qAnneal,
                               const EigenRowVector &h, const EigenMatrix &J,
                               const EigenMatrix &qFixed, real G, real beta) {
#ifdef _OPENMP    
    real twoDivM = real(2.) / m_;
    real coef = std::log(std::tanh(G * beta / m_)) * beta;

    int m2 = (m_ / 2) * 2; /* round down */
    EigenMatrix dEmat(qFixed.rows(), J.rows());
    // dEmat = qFixed * J.transpose();  // For debug
#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        int qRowSpan = (qFixed.rows() + nMaxThreads_ - 1) / nMaxThreads_;
        int qRowBegin = std::min(J.rows(), qRowSpan * threadNum);
        int qRowEnd = std::min(qFixed.rows(), qRowSpan * (threadNum + 1));
        qRowSpan = qRowEnd - qRowBegin;
        if (0 < qRowSpan)
            dEmat.block(qRowBegin, 0, qRowSpan, J.rows()) = qFixed.block(qRowBegin, 0, qRowSpan, qFixed.cols()) * J.transpose();
    }

#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        sq::Random &random = random_[threadNum];
#  pragma omp for
        for (int im = 0; im < m2; im += 2) {
            tryFlip(qAnneal, im, dEmat, h, J, N, m_, twoDivM, beta, coef, random);
        }
    }
    if ((m_ % 2) != 0) { /* m is odd. */
        int im = m_ - 1;
        tryFlip(qAnneal, im, dEmat, h, J, N, m_, twoDivM, beta, coef, random_[0]);
    }

#pragma omp parallel
    {    
        int threadNum = omp_get_thread_num();
        sq::Random &random = random_[threadNum];
#  pragma omp for
        for (int im = 1; im < m2; im += 2) {
            tryFlip(qAnneal, im, dEmat, h, J, N, m_, twoDivM, beta, coef, random);
        }
    }
#else
    abort_("Must not reach here."):
#endif
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepColoringParallel(real G, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);

    annealHalfStepColoringParallel(N1_, matQ1_, h1_, J_, matQ0_, G, beta);
    annealHalfStepColoringParallel(N0_, matQ0_, h0_, J_.transpose(), matQ1_, G, beta);
}



template<class real>
void CPUBipartiteGraphAnnealer<real>::syncBits() {
    bitsPairX_.clear();
    bitsPairQ_.clear();
    sq::BitSet x0, x1;
    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        sq::BitSet q0 = sq::extractRow<char>(matQ0_, idx);
        sq::BitSet q1 = sq::extractRow<char>(matQ1_, idx);
        bitsPairQ_.pushBack(sq::BitSetPairArray::ValueType(q0, q1));
        sq::BitSet x0 = x_from_q(q0), x1 = x_from_q(q1);
        bitsPairX_.pushBack(sq::BitSetPairArray::ValueType(x0, x1));
    }
}


template class CPUBipartiteGraphAnnealer<float>;
template class CPUBipartiteGraphAnnealer<double>;
