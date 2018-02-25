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
class CUDADenseGraphAnnealer : public sqaod::DenseGraphAnnealer<real> {
    typedef sqaod::MatrixType<real> HostMatrix;
    typedef sqaod::VectorType<real> HostVector;
    typedef sqaod::Bits Bits;
    typedef sqaod::BitsArray BitsArray;
    typedef sqaod::SizeType SizeType;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceDenseGraphFormulas<real> DeviceFormulas;
    
public:
    CUDADenseGraphAnnealer();

    CUDADenseGraphAnnealer(Device &device);

    ~CUDADenseGraphAnnealer();

    void deallocate();

    void assignDevice(Device &device);

    virtual Algorithm selectAlgorithm(Algorithm algo);
    
    virtual Algorithm getAlgorithm() const;
    
    void seed(unsigned int seed);

    void setProblem(const HostMatrix &W, sqaod::OptimizeMethod om = sqaod::optMinimize);

    /* void getProblemSize(sqaod::SizeType *N) const; */

    /* Preferences getPreferences() const; */

    /* void setPreference(const Preference &pref); */
    
    const HostVector &get_E() const;

    const BitsArray &get_x() const;

    void set_x(const Bits &x);

    const sqaod::BitsArray &get_q() const {
        return qlist_;
    }

    void get_hJc(HostVector *h, HostMatrix *J, real *c) const;

    void randomize_q();

    void calculate_E();

    void initAnneal();

    void finAnneal();

    void annealOneStep(real G, real kT);

    /* CUDA Kernels */
    void annealOneStep(DeviceMatrix *d_matq, const DeviceVector &d_Jq,
                       const int *d_x, const real *d_random,
                       const DeviceVector &d_h, const DeviceMatrix &d_J, real G, real kT);

    void calculate_Jq(DeviceVector *d_Jq, const DeviceMatrix &d_J, const DeviceMatrix &d_matq,
                      const int *d_flipPos);
private:
    void deallocateProblem();
    void deallocateInternalObjects();

    enum {
        /* FIXME: parameterise */
        nRunsPerRandGen = 10
    };

    void syncBits();

    DeviceRandom d_random_;
    DeviceMatrix d_J_;
    DeviceVector d_h_;
    DeviceScalar d_c_;
    DeviceMatrix d_matq_;
    DeviceVector d_Jq_;
    DeviceVector h_E_;
    DeviceBitMatrix h_q_;
    DeviceRandomBuffer flipPosBuffer_;
    DeviceRandomBuffer realNumBuffer_;
    HostVector E_;

    sqaod::BitsArray xlist_;
    sqaod::BitsArray qlist_;

    DeviceSegmentedSumType<real> *dotJq_;

    DeviceStream *devStream_;
    DeviceFormulas devFormulas_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;

    typedef sqaod::DenseGraphAnnealer<real> Base;
    using Base::N_;
    using Base::m_;
    using Base::om_;
    /* annealer state */
    using Base::solRandSeedGiven;
    using Base::solInitialized;
    using Base::solProblemSet;
    using Base::solQSet;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::clearState;
    using Base::isRandSeedGiven;
    using Base::isProblemSet;
    using Base::isInitialized;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotInitialized;
    using Base::throwErrorIfQNotSet;
    using Base::throwErrorIfSolutionNotAvailable;
};

}
