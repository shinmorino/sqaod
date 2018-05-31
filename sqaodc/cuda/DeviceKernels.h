#pragma once

#include <sqaodc/cuda/DeviceStream.h>
#include <sqaodc/cuda/DeviceRandom.h>
#include <sqaodc/cuda/DeviceMatrix.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
struct DeviceMathKernelsType {

    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;

    void scale(DeviceScalar *d_y, real alpha, const DeviceScalar &d_x, real addAssignFactor);

    void scale(DeviceVector *d_y, real alpha, const DeviceVector &d_x, real addAssignFactor);

    void scale(DeviceMatrix *d_A, real alpha, const DeviceMatrix &d_X, real addAssignFactor);
    
    void scaleBroadcast(DeviceVector *d_x, real alpha, const DeviceScalar &d_c, real addAssignFactor);

    void scaleBroadcast(DeviceMatrix *d_A, real alpha, const DeviceScalar &d_c, real addAssignFactor);
    
    void scaleBroadcastToRows(DeviceMatrix *d_A, 
                              real alpha, const DeviceVector &d_x, real addAssignFactor);

    void scaleBroadcastToColumns(DeviceMatrix *d_A,
                                 real alpha, const DeviceVector &d_x, real addAssignFactor);

    void sum(DeviceScalar *d_dst, real alpha, const DeviceVector &d_x, real addAssignFactor);

    void sum(DeviceScalar *d_dst, real alpha, const DeviceMatrix &d_A, real addAssignFactor);
    
    void sumDiagonal(DeviceScalar *d_dst, real alpha, const DeviceMatrix &d_A, sq::SizeType offset, real addAssignFactor);
    
    void sumRowwise(DeviceVector *d_x, real alpha, const DeviceMatrix &d_A);

    void dot(DeviceScalar *d_c, real alpha, const DeviceVector &d_x, const DeviceVector &d_y, real addAssignFactor);
    
    void dotRowwise(DeviceVector *d_z,
                    real alpha, const DeviceMatrix &d_X, const DeviceMatrix &d_Y);

    void transpose(DeviceMatrix *d_tr, const DeviceMatrix &d_mat);

    void min(DeviceScalar *d_min, const DeviceVector &d_x);

    void min(DeviceScalar *d_min, const DeviceMatrix &d_A);

    void gemv(cublasOperation_t op, int M, int N,
              const real *d_alpha, const real *d_A, sq::SizeType Astride,
              const real *d_x, const real *d_beta, real *d_y);
 
    void gemm(cublasOperation_t opA, cublasOperation_t opB, int M, int N, int K,
              const real *d_alpha,
              const real *d_A, int lda,
              const real *d_B, int ldb,
              const real *d_beta, real *d_C, int ldc);

    DeviceMathKernelsType(DeviceStream *devStream = NULL);
    
    void assignStream(DeviceStream *devStream);
    
private:
    cudaStream_t stream_;
    DeviceStream *devStream_;

    sq::NullBase *segmentedSum_;
    sq::NullBase *segmentedDot_;
};


struct DeviceCopyKernels {

    template<class V>
    void broadcast(DeviceVectorType<V> *dst, const V &v) const;

    template<class V>
    void broadcast(DeviceMatrixType<V> *dst, const V &v) const;

    template<class V>
    void broadcastToRows(DeviceMatrixType<V> *dst, const DeviceVectorType<V> &vec) const;

    template<class V>
    void broadcastToDiagonal(DeviceMatrixType<V> *d_A, const V &v, sq::IdxType offset) const;

    template<class Vdst, class Vsrc>
    void cast(DeviceVectorType<Vdst> *dst, const DeviceVectorType<Vsrc> &src);

    template<class Vdst, class Vsrc>
    void cast(DeviceMatrixType<Vdst> *dst, const DeviceMatrixType<Vsrc> &src);

    DeviceCopyKernels(DeviceStream *stream = NULL);

    void assignStream(DeviceStream *stream);

private:
    cudaStream_t stream_;
};

template<class V>
void generateBitSetSequence(DeviceMatrixType<V> *d_data, 
                            sq::PackedBitSet xBegin, sq::PackedBitSet xEnd,
                            cudaStream_t stream);

template<class V>
void randomizeSpin(DeviceVectorType<V> *d_matq, DeviceRandom &d_random, cudaStream_t stream);

template<class V>
void randomizeSpin(DeviceMatrixType<V> *d_matq, DeviceRandom &d_random, cudaStream_t stream);

}  // namespace sqaod_cuda
