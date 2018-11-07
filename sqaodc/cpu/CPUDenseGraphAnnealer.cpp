#include "CPUDenseGraphAnnealer.h"
#include "SharedFormulas.h"
#include <common/Common.h>
#include <sqaodc/common/EigenBridge.h>
#include <sqaodc/common/internal/ShapeChecker.h>
#include <time.h>
#include <omp.h>
#include "Dot_SIMD.h"

namespace sqint = sqaod_internal;
using namespace sqaod_cpu;

template<class real>
CPUDenseGraphAnnealer<real>::CPUDenseGraphAnnealer() {
    m_ = -1;
    selectAlgorithm(sq::algoDefault);
    nWorkers_ = sq::getDefaultNumThreads();
    sq::log("# workers: %d", nWorkers_);
    random_ = new sq::Random[nWorkers_];
}

template<class real>
CPUDenseGraphAnnealer<real>::~CPUDenseGraphAnnealer() {
    delete [] random_;
}

template<class real>
void CPUDenseGraphAnnealer<real>::seed(unsigned long long seed) {
    for (int idx = 0; idx < nWorkers_; ++idx)
        random_[idx].seed(seed + 17 * idx);
    setState(solRandSeedGiven);
}


template<class real>
sq::Algorithm CPUDenseGraphAnnealer<real>::selectAlgorithm(enum sq::Algorithm algo) {
    switch (algo) {
    case sq::algoNaive:
    case sq::algoColoring:
    case sq::algoSANaive:
        algo_ = algo;
        break;
    default:
        selectDefaultAlgorithm(algo, sq::algoColoring, sq::algoSANaive);
        break;
    }
    return algo_;
}

template<class real>
void CPUDenseGraphAnnealer<real>::setQUBO(const Matrix &W, sq::OptimizeMethod om) {
    sqint::quboShapeCheck(W, __func__);

    N_ = W.rows;
    m_ = N_ / 4;
    h_.resize(N_);
    J_.resize(N_, N_);
    
    DGFuncs<real>::calculateHamiltonian(&h_, &J_, &c_, W);
    J_.clearPadding();
    om_ = om;
    if (om_ == sq::optMaximize) {
        h_ *= real(-1.);
        J_ *= real(-1.);
        c_ *= real(-1.);
    }
    setState(solProblemSet);
}

template<class real>
void CPUDenseGraphAnnealer<real>::setHamiltonian(const Vector &h, const Matrix &J, real c) {
    sqint::isingModelShapeCheck(h, J, c, __func__);

    N_ = J.rows;
    m_ = N_ / 4;

    om_ = sq::optMinimize;
    h_ = h;
    J_ = J;
    c_ = c;
    J_.clearPadding();
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
    if (!isEAvailable())
        const_cast<This*>(this)->calculate_E();
    return E_;
}

template<class real>
const sq::BitSetArray &CPUDenseGraphAnnealer<real>::get_x() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return bitsX_;
}

template<class real>
void CPUDenseGraphAnnealer<real>::set_q(const sq::BitSet &q) {
    sqint::isingModelShapeCheck(h_, J_, c_, q, __func__);
    throwErrorIfNotPrepared();
    throwErrorIf(q.size != N_,
                 "Dimension of q, %d, should be equal to N, %d.", q.size, N_);
    sq::EigenMappedMatrixType<real> matQ(mapTo(matQ_));
    sq::EigenMappedRowVectorType<char> eq(mapToRowVector(q));
    for (int idx = 0; idx < m_; ++idx)
        matQ.row(idx) = eq.cast<real>();
    matQ_.clearPadding();
    setState(solQSet);
}

template<class real>
void CPUDenseGraphAnnealer<real>::set_qset(const sq::BitSetArray &q) {
    sqint::isingModelShapeCheck(h_, J_, c_, q, __func__);
    m_ = q.size();
    prepare(); /* update num trotters */
    sq::EigenMappedMatrixType<real> matQ(mapTo(matQ_));
    for (int idx = 0; idx < m_; ++idx) {
        Vector qvec = sq::cast<real>(q[idx]);
        matQ.row(idx) = sq::mapToRowVector(qvec);
    }
    matQ_.clearPadding();
    setState(solQSet);
}

template<class real>
void CPUDenseGraphAnnealer<real>::getHamiltonian(Vector *h, Matrix *J, real *c) const {
    throwErrorIfProblemNotSet();
    *h = h_;
    *J = J_;
    *c = c_;
}

template<class real>
const sq::BitSetArray &CPUDenseGraphAnnealer<real>::get_q() const {
    if (!isSolutionAvailable())
        const_cast<This*>(this)->makeSolution();
    return bitsQ_;
}

template<class real>
void CPUDenseGraphAnnealer<real>::randomizeSpin() {
    throwErrorIfNotPrepared();
    for (int row = 0; row < m_; ++row) {
        real *q = matQ_.rowPtr(row);
        for (int col = 0; col < N_; ++col)
            q[col] = random_->randInt(2) ? real(1.) : real(-1.);
    }
    matQ_.clearPadding();
    setState(solQSet);
}

template<class real>
void CPUDenseGraphAnnealer<real>::prepare() {
    if (!isRandSeedGiven())
        seed((unsigned long long)time(NULL));

    setState(solRandSeedGiven);
    bitsX_.reserve(m_);
    bitsQ_.reserve(m_);
    matQ_.resize(m_, N_);;
    E_.resize(m_);

    if (m_ == 1)
        selectDefaultSAAlgorithm(algo_, sq::algoSANaive);
    
    switch (algo_) {
    case sq::algoNaive:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepNaive;
        break;
    case sq::algoColoring:
        if (nWorkers_ == 1)
            annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoring;
        else
            annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepColoringParallel;
        break;
    case sq::algoSANaive:
        annealMethod_ = &CPUDenseGraphAnnealer::annealOneStepSANaive;
        break;
    default:
        abort_("Must not reach here.");
    }

    setState(solPrepared);
}

template<class real>
void CPUDenseGraphAnnealer<real>::makeSolution() {
    throwErrorIfQNotSet();
    syncBits();
    setState(solSolutionAvailable);
    calculate_E();
}


template<class real>
void CPUDenseGraphAnnealer<real>::calculate_E() {
    throwErrorIfQNotSet();
    DGFuncs<real>::calculate_E(&E_, h_, J_, c_, matQ_);
    if (om_ == sq::optMaximize)
        mapToRowVector(E_) *= real(-1.);
    setState(solEAvailable);
}


template<class real>
void CPUDenseGraphAnnealer<real>::syncBits() {
    bitsX_.clear();
    bitsQ_.clear();
    
    sq::BitMatrix matBitQ = sq::cast<char>(matQ_);
    for (int idx = 0; idx < sq::IdxType(m_); ++idx) {
        sq::BitSet q(matBitQ.rowPtr(idx), matBitQ.cols);
        bitsQ_.pushBack(q);
        bitsX_.pushBack(x_from_q(q));
    }
}


template<class real> inline static
void tryFlip(sq::MatrixType<real> &matQ, int y, const sq::VectorType<real> &h, const sq::MatrixType<real> &J, 
             sq::Random &random, real twoDivM, real coef, real beta) {
    int N = J.rows;
    int m = matQ.rows;
    int x = random.randInt(N);
    real qyx = matQ(y, x);
#if defined(__AVX2__)
    real sum = dot_avx2(J.rowPtr(x), matQ.rowPtr(y), N);
#elif defined(__SSE2__)
    real sum = dot_sse2(J.rowPtr(x), matQ.rowPtr(y), N);
#else
    real sum = dot_naive(J.rowPtr(x), matQ.rowPtr(y), N);
#endif
    real dE = twoDivM * qyx * (h(x) + sum);
    int neibour0 = (y == 0) ? m - 1 : y - 1;
    int neibour1 = (y == m - 1) ? 0 : y + 1;
    dE -= qyx * (matQ(neibour0, x) + matQ(neibour1, x)) * coef;
    real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * beta);
    if (threshold > random.random<real>())
        matQ(y, x) = - qyx;
}


template<class real>
void CPUDenseGraphAnnealer<real>::annealOneStepNaive(real G, real beta) {
    throwErrorIfQNotSet();

    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G * beta / m_)) * beta;
    sq::Random &random = random_[0];
    for (int loop = 0; loop < sq::IdxType(N_ * m_); ++loop) {
        int y = random.randInt(m_);
        tryFlip(matQ_, y, h_, J_, random, twoDivM, coef, beta);
    }
    clearState(solSolutionAvailable);
}

template<class real>
void CPUDenseGraphAnnealer<real>::annealColoredPlane(real G, real beta) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G * beta / m_)) * beta;
    /* single thread */
    sq::Random &random = random_[0];
    for (int yOffset = 0; yOffset < 2; ++yOffset) {
        for (int y = yOffset; y < m_; y += 2)
            tryFlip(matQ_, y, h_, J_, random, twoDivM, coef, beta);
    }
}

template<class real>
void CPUDenseGraphAnnealer<real>::annealOneStepColoring(real G, real beta) {
    throwErrorIfQNotSet();
    
    for (int idx = 0; idx < (sq::IdxType)N_; ++idx)
        annealColoredPlane(G, beta);
    clearState(solSolutionAvailable);
}


template<class real>
void CPUDenseGraphAnnealer<real>::annealColoredPlaneParallel(real G, real beta) {
#ifdef _OPENMP
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G * beta / m_)) * beta;

    sq::IdxType m2 = (m_ / 2) * 2; /* round down */
#  pragma omp parallel
    {
        sq::Random &random = random_[omp_get_thread_num()];
#  pragma omp for
        for (int y = 0; y < m2; y += 2)
            tryFlip(matQ_, y, h_, J_, random, twoDivM, coef, beta);
    }

    if ((m_ % 2) != 0) /* m is odd. */
        tryFlip(matQ_, m_ - 1, h_, J_, random_[0], twoDivM, coef, beta);

#  pragma omp parallel
    {
        sq::Random &random = random_[omp_get_thread_num()];
#  pragma omp for
        for (int y = 1; y < m2; y += 2)
            tryFlip(matQ_, y, h_, J_, random, twoDivM, coef, beta);
    }
#endif
}

template<class real>
void CPUDenseGraphAnnealer<real>::annealOneStepColoringParallel(real G, real beta) {
    throwErrorIfQNotSet();
    
    for (int idx = 0; idx < (sq::IdxType)N_; ++idx)
        annealColoredPlaneParallel(G, beta);
    clearState(solSolutionAvailable);
}


template<class real> inline static
void tryFlipSA(sq::MatrixType<real> &matQ, int y, const sq::VectorType<real> &h, const sq::MatrixType<real> &J, 
             sq::Random &random, real invKT) {
    int N = J.rows;
    int x = random.randInt(N);
    real qyx = matQ(y, x);
#if defined(__AVX2__)
    real sum = dot_avx2(J.rowPtr(x), matQ.rowPtr(y), N);
#elif defined(__SSE2__)
    real sum = dot_sse2(J.rowPtr(x), matQ.rowPtr(y), N);
#else
    real sum = dot_naive(J.rowPtr(x), matQ.rowPtr(y), N);
#endif
    real dE = real(2.) * qyx * (h(x) + 2. * sum);
    real threshold = (dE < real(0.)) ? real(1.) : std::exp(-dE * invKT);
    if (threshold > random.random<real>())
        matQ(y, x) = - qyx;
}

template<class real>
void CPUDenseGraphAnnealer<real>::annealOneStepSANaive(real kT, real _) {
    throwErrorIfQNotSet();

    real invKT = real(1.) / kT;
    
#ifndef _OPENMP
    sq::Random &random = random_[0];
#else
    sq::Random &random = random_[omp_get_thread_num()];
#pragma omp parallel for
#endif   
    for (int y = 0; y < m_; ++y) {
        for (int loop = 0; loop < sq::IdxType(N_); ++loop)
            tryFlipSA(matQ_, y, h_, J_, random, invKT);
    }
    clearState(solSolutionAvailable);
}


template class CPUDenseGraphAnnealer<float>;
template class CPUDenseGraphAnnealer<double>;
