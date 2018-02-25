#pragma once

#include <sqaodc/cuda/DeviceStream.h>
#include <sqaodc/cuda/DeviceRandom.h>
#include <sqaodc/cuda/DeviceSegmentedSum.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
struct DeviceMathKernelsType {
    typedef sq::SizeType SizeType;

    void scale(real *d_y, real alpha, const real *d_x, SizeType size, real addAssignFactor);
    void scaleBroadcast(real *d_x, real alpha, const real *d_c, SizeType size, real addAssignFactor);
    void scaleBroadcastVector(real *d_A, real alpha, const real *d_x, SizeType size,
                              SizeType nBatch, real addAssignFactor);
    void scaleBroadcastScalars(real *d_A, real alpha, const real *d_x, SizeType size,
                               SizeType nBatch, real addAssignFactor);

    void sum(real *d_dst, real alpha, const real *d_x, SizeType size, real addAssignFactor);
    void sumGather(real *d_dst, real alpha, const real *d_x, SizeType size, SizeType stride, int offset);
    void sumBatched(real *d_x, real alpha, const real *d_A, SizeType size, SizeType nBatch);
    void dot(real *d_c, real alpha, const real *d_x, const real *d_y, SizeType size,
             real addAssignFactor);
    void dotBatched(real *d_z, real alpha, const real *d_x, const real *d_y, SizeType size,
                    SizeType nBatch);

    void transpose(real *d_tr, const real *d_mat, SizeType rows, SizeType cols);

    void min(real *d_min, const real *d_values, SizeType size);

    void gemv(cublasOperation_t op, int M, int N,
        const real *d_alpha, const real *d_A, const real *d_x,
        const real *d_beta, real *d_y);
 
    void gemm(cublasOperation_t opA, cublasOperation_t opB, int M, int N, int K,
              const real *d_alpha, const real *d_A, int lda, const real *d_B, int ldb,
              const real *d_beta, real *d_C, int ldc);

    void toBits(char *bits, const real *values, sq::SizeType size);

    DeviceMathKernelsType(DeviceStream *devStream = NULL);

    void assignStream(DeviceStream *devStream);

private:
    cudaStream_t stream_;
    DeviceStream *devStream_;

    DeviceSegmentedSumType<real> *segmentedSum_;
    DeviceSegmentedSumType<real> *segmentedDot_;
};


struct DeviceCopyKernels {

    template<class V>
    void copyBroadcast(V *d_buf, const V &v, sq::SizeType nElms) const;

    template<class V>
    void copyBroadcastStrided(V *d_buf, const V &v, sq::SizeType size,
                              sq::SizeType stride, sq::IdxType offset) const;

    DeviceCopyKernels(DeviceStream *stream = NULL);

    void assignStream(DeviceStream *stream);

private:
    cudaStream_t stream_;
};



template<class V>
void generateBitsSequence(V *d_data, int N,
                          sq::PackedBits xBegin, sq::PackedBits xEnd,
                          cudaStream_t stream);


template<class V>
void randomize_q(V *d_matq, DeviceRandom &d_random, sq::SizeType size, cudaStream_t stream);

}  // namespace sqaod_cuda
