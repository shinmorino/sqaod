#pragma once

#include <sqaodc/cuda/DeviceStream.h>
#include <sqaodc/cuda/DeviceRandom.h>
#include <sqaodc/cuda/DeviceSegmentedSum.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
struct DeviceMathKernelsType {

    void scale(real *d_y, real alpha, const real *d_x, sq::SizeType size, real addAssignFactor);
    
    void scale2d(real *d_y, sq::SizeType yStride,
                 real alpha, const real *d_x, sq::SizeType xStride,
                 sq::SizeType rows, sq::SizeType cols, real addAssignFactor);
    
    void scaleBroadcast(real *d_x, real alpha, const real *d_c, sq::SizeType size,
                        real addAssignFactor);

    void scaleBroadcast2d(real *d_x, sq::SizeType xStride,
                          real alpha, const real *d_c, sq::SizeType rows, sq::SizeType cols,
                          real addAssignFactor);
    
    void scaleBroadcastVector(real *d_A, sq::SizeType Astride,
                              real alpha, const real *d_x,
                              sq::SizeType size, sq::SizeType nBatch, real addAssignFactor);
    void scaleBroadcastScalars(real *d_A, sq::SizeType Astride,
                               real alpha, const real *d_x,
                               sq::SizeType size, sq::SizeType nBatch, real addAssignFactor);

    void sum(real *d_dst, real alpha, const real *d_x, sq::SizeType size, real addAssignFactor);

    void sum2d(real *d_dst, real alpha, const real *d_x, sq::SizeType xStride,
               sq::SizeType rows, sq::SizeType cols, real addAssignFactor);
    
    void sumWithInterval(real *d_dst, real alpha, const real *d_x,
                         sq::SizeType interval, sq::SizeType offset, sq::SizeType size);
    
    void sumBatched(real *d_x, real alpha, const real *d_A, sq::SizeType stride,
                    sq::SizeType size, sq::SizeType nBatch);
    void dot(real *d_c, real alpha, const real *d_x, const real *d_y, sq::SizeType size,
             real addAssignFactor);
    
    void dotBatched(real *d_z,
                    real alpha, const real *d_x, sq::SizeType xStride,
                    const real *d_y, sq::SizeType yStride,
                    sq::SizeType size, sq::SizeType nBatch);

    void transpose2d(real *d_tr, sq::SizeType trStride,
                     const real *d_mat, sq::SizeType matStride,
                     sq::SizeType matRows, sq::SizeType matCols);

    void min(real *d_min,
             const real *d_values, sq::SizeType size);

    void min2d(real *d_min,
               const real *d_values, sq::SizeType stride, sq::SizeType rows, sq::SizeType cols);

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

    DeviceSegmentedSumType<real> *segmentedSum_;
    DeviceSegmentedSumType<real> *segmentedDot_;
};


struct DeviceCopyKernels {

    template<class V>
    void copyBroadcast(V *d_buf, const V &v, sq::SizeType size) const;

    template<class V>
    void copyBroadcast2d(V *d_buf, sq::SizeType stride, const V &v,
                         sq::SizeType rows, sq::SizeType cols) const;

    template<class V>
    void broadcastToDiagonal(V *d_buf, sq::SizeType stride, const V &v, sq::SizeType width, sq::SizeType height, sq::IdxType offset) const;

    template<class V>
    void copyBroadcastVector(V *dst, sq::SizeType dstStride,
                             const V *vec, sq::SizeType size, sq::SizeType nBatch) const;

    template<class Vdst, class Vsrc>
    void cast(Vdst *dst, const Vsrc *src, sq::SizeType size);

    template<class Vdst, class Vsrc>
    void cast2d(Vdst *dst, sq::SizeType dstStride, const Vsrc *src, sq::SizeType srcStride,
                sq::SizeType rows, sq::SizeType cols);

    DeviceCopyKernels(DeviceStream *stream = NULL);

    void assignStream(DeviceStream *stream);

private:
    cudaStream_t stream_;
};



template<class V>
void generateBitsSequence(V *d_data, int N,
                          sq::PackedBitSet xBegin, sq::PackedBitSet xEnd,
                          cudaStream_t stream);


template<class V>
void randomizeSpin(V *d_matq, DeviceRandom &d_random, sq::SizeType size, cudaStream_t stream);

template<class V>
void randomizeSpin2d(V *d_matq, sq::SizeType stride, DeviceRandom &d_random, sq::SizeType rows, sq::SizeType cols, cudaStream_t stream);

}  // namespace sqaod_cuda
