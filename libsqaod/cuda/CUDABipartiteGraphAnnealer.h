/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <cuda/Device.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceDenseGraphBatchSearch.h>
#include <cuda/DeviceRandom.h>
#include <cuda/DeviceRandomBuffer.h>


namespace sqaod_cuda {

template<class real>
class CUDABipartiteGraphAnnealer : public sqaod::BipartiteGraphAnnealer<real> {
    typedef sqaod::MatrixType<real> HostMatrix;
    typedef sqaod::VectorType<real> HostVector;
    typedef sqaod::BitsPairArray BitsPairArray;
    typedef sqaod::Bits Bits;
    typedef sqaod::SizeType SizeType;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceBipartiteGraphFormulas<real> DeviceFormulas;

    typedef sqaod::BipartiteGraphAnnealer<real> Base;

    using Base::annState_;
    using Base::om_;
    using Base::N0_;
    using Base::N1_;
    using Base::m_;
    
public:
    CUDABipartiteGraphAnnealer();

    CUDABipartiteGraphAnnealer(Device &device);

    ~CUDABipartiteGraphAnnealer();

    void deallocate();
    
    void assignDevice(Device &device);

    virtual Algorithm selectAlgorithm(Algorithm algo);
    
    virtual Algorithm getAlgorithm() const;
    
    void seed(unsigned int seed);

    /* void getProblemSize(SizeType *N0, SizeType *N1) const; */

    void setProblem(const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                    sqaod::OptimizeMethod om = sqaod::optMinimize);

    /* Preferences getPreferences() const; */

    /* void setPreference(const Preference &pref); */

    const HostVector &get_E() const {
        return E_;
    }

    const BitsPairArray &get_x() const;

    void set_x(const Bits &x0, const Bits &x1);

    /* Ising machine / spins */

    void get_hJc(HostVector *h0, HostVector *h1, HostMatrix *J, real *c) const;

    const BitsPairArray &get_q() const;

    void randomize_q();

    void calculate_E();

    void initAnneal();

    void finAnneal();

    void annealOneStep(real G, real kT);
    

    /* public for debug */
    void calculate_Jq(DeviceMatrix *d_Jq, const DeviceMatrix &d_J, MatrixOp op,
                      const DeviceMatrix &d_qFixed);

    /* public for debug */
    void tryFlip(DeviceMatrix *d_qAnneal, const DeviceMatrix &d_Jq, int N, int m, 
                 int nTrottersToFlipe, int offset,
                 const DeviceVector &d_h, const real *d_realRand, real G, real kT);

private:
    enum {
        /* FIXME: parameterise */
        nRunsPerRandGen = 10
    };

    void syncBits();

    DeviceRandom d_random_;
    DeviceRandomBuffer d_randReal_;
    DeviceVector d_h0_, d_h1_;
    DeviceMatrix d_J_;
    DeviceScalar d_c_;
    DeviceMatrix d_matq0_, d_matq1_;
    DeviceBitMatrix h_q0_, h_q1_;
    
    DeviceMatrix d_Jq0_;
    DeviceMatrix d_Jq1_;
    
    DeviceVector h_E_; /* host mem */
    HostVector E_;
    BitsPairArray bitsPairX_;
    BitsPairArray bitsPairQ_;
    
    DeviceStream *devStream_;
    DeviceFormulas devFormulas_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;
};

}
