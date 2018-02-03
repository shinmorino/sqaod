#pragma once

#include <common/Common.h>
#include <cuda/CUDAFormulas.h>
#include <cuda/DeviceRandom.h>
#include <cuda/DeviceRandomBuffer.h>


namespace sqaod_cuda {


template<class real>
class CUDADenseGraphAnnealer {
typedef sqaod::MatrixType<real> Matrix;
typedef sqaod::VectorType<real> Vector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
public:
    CUDADenseGraphAnnealer();

    CUDADenseGraphAnnealer(Device &device);

    ~CUDADenseGraphAnnealer();

    void assignDevice(Device &device);

    void seed(unsigned long seed);

    void setProblem(const Matrix &W, sqaod::OptimizeMethod om = sqaod::optMinimize);

    void getProblemSize(int *N, int *m) const;

    void setNumTrotters(int m);

    const Vector get_E() const {
        return Vector(h_E_.d_data, h_E_.size);
    }

    const sqaod::BitsArray &get_x() const;

    const sqaod::BitsArray &get_q() const;

    void get_hJc(Vector *h, Matrix *J, real *c) const;

    void randomize_q();

    void calculate_E();

    void initAnneal();

    void finAnneal();

    void annealOneStep(real G, real kT);

    /* CUDA Kernels */
    void annealOneStep(DeviceMatrix *d_matq, const DeviceVector &d_Jq,
                       const int *d_x, const real *d_random,
                       const DeviceVector &d_h, const DeviceMatrix &d_J, real G, real kT);

    void calculate_Jq(DeviceVector *d_E, const DeviceMatrix &J, const DeviceMatrix &d_matq,
                      const int *d_flipPos);
private:
    enum {
        nRunsPerRandGen = 100
    };

    void allocate(int N, int m);
    void deallocate();
    void syncBits();
    void calculate_Jq();

    int annState_;

    int N_, m_;
    sqaod::OptimizeMethod om_;

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
    Vector E_;

    sqaod::BitsArray xlist_;
    sqaod::BitsArray qlist_;

    int nThreadsToFillDevice_;
    DeviceStream *devStream_;
    CUDADGFuncs<real> dgFuncs_;
    DeviceMathType<real> devMath_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;
};

}
