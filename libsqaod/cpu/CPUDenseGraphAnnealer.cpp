#include "CPUDenseGraphAnnealer.h"


namespace qd = quantd_cpu;


template<class real>
qd::CPUDenseGraphAnnealer<real>::CPUDenseGraphAnnealer() {
}

template<class real>
qd::CPUDenseGraphAnnealer<real>::~CPUDenseGraphAnnealer() {
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::seed(unsigned long seed) {
    random_.seed(seed);
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::setProblemSize(int N, int m) {
    N_ = N;
    m_ = m;
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::getProblemSize(int *N, int *m) const {
    *N = N_;
    *m = m_;
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::setProblem(const real *W, OptimizeMethod om) {
    bitQ_.resize(m_, N_);
    matQ_.resize(m_, N_);;
    h_.resize(1, N_);
    J_.resize(N_, N_);
    E_.resize(m_);

    DGFuncs<real>::calculate_hJc(h_.data(), J_.data(), &c_, W, N_);
    om_ = om;
    if (om_ == qd::optMaximize) {
        h_ *= real(-1.);
        J_ *= real(-1.);
        c_ *= real(-1.);
    }
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::randomize_q() {
    real *q = matQ_.data();
    for (int idx = 0; idx < N_ * m_; ++idx)
        q[idx] = random_.randInt(2) ? real(1.) : real(-1.);
}

template<class real>
const char *qd::CPUDenseGraphAnnealer<real>::get_q() const {
    char *bq = bitQ_.data();
    const real *mq = matQ_.data();
    for (int idx = 0; idx < N_ * m_; ++idx)
        bq[idx] = (char)mq[idx];
    return bq;
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::get_hJc(const real **h, const real **J, real *c) const {
    *h = h_.data();
    *J = J_.data();
    *c = c_;
}

template<class real>
const real *qd::CPUDenseGraphAnnealer<real>::get_E() const {
    return E_.data();
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::calculate_E() {
    DGFuncs<real>::batchCalculate_E_fromQbits(E_.data(),
                                              h_.data(), J_.data(), c_, matQ_.data(),
                                              N_, m_);
}

template<class real>
void qd::CPUDenseGraphAnnealer<real>::annealOneStep(real G, real kT) {
    real twoDivM = real(2.) / real(m_);
    real coef = std::log(std::tanh(G / kT / m_)) / kT;
        
    for (int loop = 0; loop < N_ * m_; ++loop) {
        int x = random_.randInt(N_);
        int y = random_.randInt(m_);
        real qyx = matQ_(y, x);
        real sum = J_(x) * matQ_(y);
        real dE = - twoDivM * qyx * (h_(x) + sum);
        int neibour0 = (m_ + y - 1) % m_;
        int neibour1 = (y + 1) % m_;
        dE -= qyx * (matQ_(neibour0, x) + matQ_(neibour1, x)) * coef;
        real threshold = (dE < real(0.)) ? 1. : std::exp(-dE / kT);
        if (threshold > random_.random<real>())
            matQ_(y, x) = - qyx;
    }
    if (om_ == qd::optMaximize)
        E_ = - E_;
}


template class qd::CPUDenseGraphAnnealer<float>;
template class qd::CPUDenseGraphAnnealer<double>;
