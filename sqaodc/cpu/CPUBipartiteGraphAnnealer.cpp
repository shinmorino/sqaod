#include "CPUBipartiteGraphAnnealer.h"
#include <sqaodc/common/internal/ShapeChecker.h>
#include <cmath>
#include <float.h>
#include <algorithm>
#include <exception>
#include "SharedFormulas.h"
#include <time.h>
#include "Dot_SIMD.h"
#include <omp.h>

namespace sqint = sqaod_internal;
using namespace sqaod_cpu;

template<class real>
CPUBipartiteGraphAnnealer<real>::CPUBipartiteGraphAnnealer() {
    m_ = -1;
    selectAlgorithm(sq::algoDefault);
    nWorkers_ = sq::getDefaultNumThreads();
    sq::log("# workers: %d", nWorkers_);
    random_ = new sq::Random[nWorkers_];
}

template<class real>
CPUBipartiteGraphAnnealer<real>::~CPUBipartiteGraphAnnealer() {
    delete [] random_;
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::seed(unsigned long long seed) {
    for (int idx = 0; idx < nWorkers_; ++idx)
        random_[idx].seed(seed + 17 * idx);
    setState(solRandSeedGiven);
}

template<class real>
sq::Algorithm CPUBipartiteGraphAnnealer<real>::selectAlgorithm(sq::Algorithm algo) {
    switch (algo) {
    case sq::algoNaive:
    case sq::algoColoring:
    case sq::algoSANaive:
    case sq::algoSAColoring:
        algo_ = algo;
        break;
    default:
        selectDefaultAlgorithm(algo, sq::algoColoring, sq::algoSAColoring);
        break;
    }
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
    BGFuncs<real>::calculateHamiltonian(&h0_, &h1_, &J_, &c_, b0, b1, W);
    
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

    h0_ = h0;
    h1_ = h1;
    J_ = J;
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
    sqint::isingModelShapeCheck(h0_, h1_, J_, c_,
                                qPair.bits0, qPair.bits1, __func__);
    throwErrorIfNotPrepared();
    throwErrorIf(qPair.bits0.size != N0_,
                 "Dimension of q0, %d,  should be equal to N0, %d.", qPair.bits0.size, N0_);
    throwErrorIf(qPair.bits1.size != N1_,
                 "Dimension of q1, %d,  should be equal to N1, %d.", qPair.bits1.size, N1_);

    Vector q0 = sq::cast<real>(qPair.bits0);
    Vector q1 = sq::cast<real>(qPair.bits1);    
    for (int idx = 0; idx < m_; ++idx) {
        Vector(matQ0_.rowPtr(idx), N0_).copyFrom(q0);
        Vector(matQ1_.rowPtr(idx), N1_).copyFrom(q1);
    }
    matQ0_.clearPadding();
    matQ1_.clearPadding();
    
    setState(solQSet);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::set_qset(const sq::BitSetPairArray &qPairs) {
    sqint::isingModelSolutionShapeCheck(N0_, N1_, qPairs, __func__);
    m_ = qPairs.size();
    prepare();
    for (int idx = 0; idx < m_; ++idx) {
        Vector(matQ0_.rowPtr(idx), N0_).copyFrom(sq::cast<real>(qPairs[idx].bits0));
        Vector(matQ1_.rowPtr(idx), N1_).copyFrom(sq::cast<real>(qPairs[idx].bits1));
    }
    matQ0_.clearPadding();
    matQ1_.clearPadding();

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
    *h0 = h0_;
    *h1 = h1_;
    *J = J_;
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
#else
#pragma omp parallel
    {
        sq::Random &random = random_[omp_get_thread_num()];
#pragma omp for
#endif
        for (int im = 0; im < m_; ++im) {
            real *q = matQ0_.rowPtr(im);
            for (int in = 0; in < N0_; ++in)
                q[in] = random.randInt(2) ? real(1.) : real(-1.);
            q = matQ1_.rowPtr(im);
            for (int in = 0; in < N1_; ++in)
                q[in] = random.randInt(2) ? real(1.) : real(-1.);
        }
    }

#if 0
    auto randomizeWorker = [this](int threadIdx) {
        sq::Random &random = random_[threadIdx];
        if (threadIdx == 0) {
            real *q = matQ0_.data();
            for (int idx = 0; idx < sq::IdxType(N0_ * m_); ++idx)
                q[idx] = random.randInt(2) ? real(1.) : real(-1.);
        }
        else if (threadIdx == 1) {
            real *q = matQ1_.data();
            for (int idx = 0; idx < sq::IdxType(N1_ * m_); ++idx)
                q[idx] = random.randInt(2) ? real(1.) : real(-1.);
        }
    };
#endif
    matQ0_.clearPadding();
    matQ1_.clearPadding();
    
    setState(solQSet);
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::calculate_E() {
    throwErrorIfQNotSet();
    BGFuncs<real>::calculate_E(&E_, h0_, h1_, J_, c_, matQ0_, matQ1_);
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

    if (m_ == 1)
        selectDefaultSAAlgorithm(algo_, sq::algoSAColoring);

    switch (algo_) {
    case sq::algoNaive:
        annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepNaive;
        break;
    case sq::algoColoring:
        if (nWorkers_ == 1)
            annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepColoring;
        else
            annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepColoringParallel;
        break;
    case sq::algoSANaive:
        annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepSANaive;
        break;
    case sq::algoSAColoring:
        annealMethod_ = &CPUBipartiteGraphAnnealer<real>::annealOneStepSAColoring;
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
real CPUBipartiteGraphAnnealer<real>::getSystemE(real G, real beta) const {
    const_cast<CPUBipartiteGraphAnnealer<real>*>(this)->calculate_E();
    real E = E_.sum() / m_;

    if (isSQAAlgorithm(algo_)) {
        real spinDotSum = real(0.);
        for (int y0 = 0; y0 < m_; ++y0) {
            int y1 = (y0 + 1) % m_;
            spinDotSum += dot_simd(matQ0_.rowPtr(y0), matQ0_.rowPtr(y1), N0_);
            spinDotSum += dot_simd(matQ1_.rowPtr(y0), matQ1_.rowPtr(y1), N1_);
        }
        real coef = real(0.5) / beta * std::log(std::tanh(G * beta / m_));
        E -= spinDotSum * coef;
    }
    if (om_ == sq::optMaximize)
        E *= real(-1.);
    return E;
}

template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepNaive(real G, real beta) {
    throwErrorIfQNotSet();
    
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G * beta / m_)) / beta;
    sq::Random &random = random_[0];
    int N = N0_ + N1_;

    sq::EigenMappedMatrixType<real> eJ(sq::mapTo(J_));
    sq::EigenMappedMatrixType<real> eMatQ0(sq::mapTo(matQ0_));
    sq::EigenMappedMatrixType<real> eMatQ1(sq::mapTo(matQ1_));
    
    for (int loop = 0; loop < sq::IdxType(N * m_); ++loop) {
        int x = random.randInt(N);
        int y = random.randInt(m_);
        if (x < N0_) {
            real qyx = matQ0_(y, x);
            real sum = eJ.transpose().row(x).dot(eMatQ1.row(y));
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
            real sum = eJ.row(x).dot(eMatQ0.row(y));
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

template<class real> static inline
void tryFlip(sq::MatrixType<real> &qAnneal, int im, const sq::MatrixType<real> &dEmat, const sq::VectorType<real> &h, sq::SizeType N, sq::SizeType m, 
             real twoDivM, real beta, real coef, sq::Random &random) {
    for (int iq = 0; iq < N; ++iq) {
        real q = qAnneal(im, iq);
        real dE = twoDivM * q * (h(iq) + dEmat(im, iq));
        int mNeibour0 = (im + m - 1) % m;
        int mNeibour1 = (im + 1) % m;
        dE -= q * (qAnneal(mNeibour0, iq) + qAnneal(mNeibour1, iq)) * coef;
        real thresh = dE < real(0.) ? real(1.) : std::exp(- dE * beta);
        if (thresh > random.random<real>())
            qAnneal(im, iq) = -q;
    }
}

template<class real>template<class T>
void CPUBipartiteGraphAnnealer<real>::
annealHalfStepColoring(int N, Matrix &qAnneal,
                       const Vector &h, const T &eJ,
                       const Matrix &qFixed, real G, real beta) {
    real twoDivM = real(2.) / m_;
    real coef = std::log(std::tanh(G * beta / m_)) / beta;
    
    Matrix dEmat(qFixed.rows, eJ.rows());
    sq::mapTo(dEmat) = sq::mapTo(qFixed) * eJ.transpose();
    
    sq::Random &random = random_[0];
    for (int offset = 0; offset < 2; ++offset) {
        for (int im = offset; im < m_; im += 2)
            tryFlip(qAnneal, im, dEmat, h, N, m_, twoDivM, beta, coef, random);
    }
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepColoring(real G, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);

    sq::EigenMappedMatrixType<real> eJ(sq::mapTo(J_));
    annealHalfStepColoring(N1_, matQ1_, h1_, eJ, matQ0_, G, beta);
    annealHalfStepColoring(N0_, matQ0_, h0_, eJ.transpose(), matQ1_, G, beta);
}


template<class real>template<class T>
void CPUBipartiteGraphAnnealer<real>::
annealHalfStepColoringParallel(int N, Matrix &qAnneal,
                               const Vector &h, const T &eJ,
                               const Matrix &qFixed, real G, real beta) {
#ifdef _OPENMP    
    real twoDivM = real(2.) / m_;
    real coef = std::log(std::tanh(G * beta / m_)) / beta;

    int m2 = (m_ / 2) * 2; /* round down */
    Matrix dEmat(qFixed.rows, eJ.rows());

    sq::EigenMappedMatrixType<real> edEmat(sq::mapTo(dEmat));
    sq::EigenMappedMatrixType<real> eqFixed(sq::mapTo(qFixed));    
    
    // dEmat = qFixed * J.transpose();  // For debug
#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        int qRowSpan = (eqFixed.rows() + nWorkers_ - 1) / nWorkers_;
        int qRowBegin = std::min(eJ.rows(), qRowSpan * threadNum);
        int qRowEnd = std::min(eqFixed.rows(), qRowSpan * (threadNum + 1));
        qRowSpan = qRowEnd - qRowBegin;
        if (0 < qRowSpan)
            edEmat.block(qRowBegin, 0, qRowSpan, eJ.rows()) =
                    eqFixed.block(qRowBegin, 0, qRowSpan, eqFixed.cols()) * eJ.transpose();
    }

#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        sq::Random &random = random_[threadNum];
#  pragma omp for
        for (int im = 0; im < m2; im += 2) {
            tryFlip(qAnneal, im, dEmat, h, N, m_, twoDivM, beta, coef, random);
        }
    }
    if ((m_ % 2) != 0) { /* m is odd. */
        int im = m_ - 1;
        tryFlip(qAnneal, im, dEmat, h, N, m_, twoDivM, beta, coef, random_[0]);
    }

#pragma omp parallel
    {    
        int threadNum = omp_get_thread_num();
        sq::Random &random = random_[threadNum];
#  pragma omp for
        for (int im = 1; im < m2; im += 2) {
            tryFlip(qAnneal, im, dEmat, h, N, m_, twoDivM, beta, coef, random);
        }
    }
#else
    abort_("Must not reach here.");
#endif
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepColoringParallel(real G, real beta) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);

    sq::EigenMappedMatrixType<real> eJ(sq::mapTo(J_));
    annealHalfStepColoringParallel(N1_, matQ1_, h1_, eJ, matQ0_, G, beta);
    annealHalfStepColoringParallel(N0_, matQ0_, h0_, eJ.transpose(), matQ1_, G, beta);
}



template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepSANaive(real kT, real _) {
    throwErrorIfQNotSet();

    real invKT = real(1.) / kT;
    
    sq::EigenMappedMatrixType<real> eJ(sq::mapTo(J_));
    sq::EigenMappedMatrixType<real> eMatQ0(sq::mapTo(matQ0_));
    sq::EigenMappedMatrixType<real> eMatQ1(sq::mapTo(matQ1_));
    
    sq::Random &random = random_[0];
    int N = N0_ + N1_;
    for (int y = 0; y < m_; ++y) {
        for (int loop = 0; loop < sq::IdxType(N * m_); ++loop) {
            int x = random.randInt(N);
            if (x < N0_) {
                real qyx = eMatQ0(y, x);
                real sum = eJ.transpose().row(x).dot(eMatQ1.row(y));
                real dE = real(2.) * qyx * (h0_(x) + sum);
                real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * invKT);
                if (threshold > random.random<real>())
                    eMatQ0(y, x) = - qyx;
            }
            else {
                x -= N0_;
                real qyx = eMatQ1(y, x);
                real sum = eJ.row(x).dot(eMatQ0.row(y));
                real dE = real(2.) * qyx * (h1_(x) + sum);
                real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * invKT);
                if (threshold > random.random<real>())
                    eMatQ1(y, x) = - qyx;
            }
        }
        clearState(solSolutionAvailable);
    }
}


/* Simulated annealing */

template<class real> static inline
void tryFlipSA(sq::MatrixType<real> &qAnneal, int im,
               const sq::MatrixType<real> &dEmat, const sq::VectorType<real> &h,
               sq::SizeType N, real invKT, sq::Random &random) {
    for (int iq = 0; iq < N; ++iq) {
        real q = qAnneal(im, iq);
        real dE = real(2.) * q * (h(iq) + dEmat(im, iq));
        real thresh = dE < real(0.) ? real(1.) : std::exp(- dE * invKT);
        if (thresh > random.random<real>())
            qAnneal(im, iq) = -q;
    }
}

template<class real>template<class T>
void CPUBipartiteGraphAnnealer<real>::
annealHalfStepSAColoring(int N, Matrix &qAnneal,
                         const Vector &h, const T &eJ,
                         const Matrix &qFixed, real invKT) {
#ifdef _OPENMP    
    Matrix dEmat(qFixed.rows, eJ.rows());
        
    sq::EigenMappedMatrixType<real> edEmat(sq::mapTo(dEmat));
    sq::EigenMappedMatrixType<real> eqFixed(sq::mapTo(qFixed));
    
    // dEmat = qFixed * J.transpose();  // For debug
#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
        int qRowSpan = (eqFixed.rows() + nWorkers_ - 1) / nWorkers_;
        int qRowBegin = std::min(eJ.rows(), qRowSpan * threadNum);
        int qRowEnd = std::min(eqFixed.rows(), qRowSpan * (threadNum + 1));
        qRowSpan = qRowEnd - qRowBegin;
        if (0 < qRowSpan) {
            edEmat.block(qRowBegin, 0, qRowSpan, eJ.rows()) =
                    eqFixed.block(qRowBegin, 0, qRowSpan, eqFixed.cols()) * eJ.transpose();
        }
#  pragma omp barrier
        sq::Random &random = random_[threadNum];
#  pragma omp for
        for (int im = 0; im < m_; ++im)
            tryFlipSA(qAnneal, im, dEmat, h, N, invKT, random);
    }
#else
    abort_("Must not reach here.");
#endif
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::annealOneStepSAColoring(real kT, real _) {
    throwErrorIfQNotSet();
    clearState(solSolutionAvailable);

    real invKT = real(1.) / kT;
    sq::EigenMappedMatrixType<real> eJ(sq::mapTo(J_));
    annealHalfStepSAColoring(N1_, matQ1_, h1_, eJ, matQ0_, invKT);
    annealHalfStepSAColoring(N0_, matQ0_, h0_, eJ.transpose(), matQ1_, invKT);
}


template<class real>
void CPUBipartiteGraphAnnealer<real>::syncBits() {
    bitsPairX_.clear();
    bitsPairQ_.clear();
    sq::BitSet x0, x1;
    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        sq::BitSet q0 = sq::cast<char>(Vector(matQ0_.rowPtr(idx), N0_));
        sq::BitSet q1 = sq::cast<char>(Vector(matQ1_.rowPtr(idx), N1_));
        bitsPairQ_.pushBack(sq::BitSetPairArray::ValueType(q0, q1));
        sq::BitSet x0 = x_from_q(q0), x1 = x_from_q(q1);
        bitsPairX_.pushBack(sq::BitSetPairArray::ValueType(x0, x1));
    }
}


template class CPUBipartiteGraphAnnealer<float>;
template class CPUBipartiteGraphAnnealer<double>;
