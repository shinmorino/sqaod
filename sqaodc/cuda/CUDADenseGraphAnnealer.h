#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/cuda/Device.h>
#include <sqaodc/cuda/DeviceMatrix.h>
#include <sqaodc/cuda/DeviceArray.h>
#include <sqaodc/cuda/DeviceDenseGraphBatchSearch.h>
#include <sqaodc/cuda/DeviceRandom.h>
#include <sqaodc/cuda/DeviceRandomBuffer.h>


namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
class CUDADenseGraphAnnealer : public sq::DenseGraphAnnealer<real> {
    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef sq::Bits Bits;
    typedef sq::BitsArray BitsArray;
    typedef sq::SizeType SizeType;
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

    virtual sq::Algorithm selectAlgorithm(sq::Algorithm algo);
    
    virtual sq::Algorithm getAlgorithm() const;
    
    void seed(unsigned int seed);

    void setProblem(const HostMatrix &W, sq::OptimizeMethod om = sq::optMinimize);

    /* void getProblemSize(sq::SizeType *N) const; */

    sq::Preferences getPreferences() const;

    /* void setPreference(const Preference &pref); */
    
    const HostVector &get_E() const;

    const BitsArray &get_x() const;

    void set_x(const Bits &x);

    const sq::BitsArray &get_q() const {
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

    sq::BitsArray xlist_;
    sq::BitsArray qlist_;

    DeviceSegmentedSumType<real> *dotJq_;

    DeviceStream *devStream_;
    DeviceFormulas devFormulas_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;

    typedef sq::DenseGraphAnnealer<real> Base;
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
