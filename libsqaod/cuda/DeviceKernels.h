#ifndef CUDA_DEVICEMATHKERNELS_H__
#define CUDA_DEVICEMATHKERNELS_H__

#include <cuda/DeviceStream.h>

namespace sqaod_cuda {

template<class real>
struct DeviceMathKernelsType {
    typedef sqaod::SizeType SizeType;
    
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

    DeviceMathKernelsType(DeviceStream *devStream = NULL);

    void assignStream(DeviceStream *devStream);

private:
    cudaStream_t stream_;
    DeviceStream *devStream_;
};


struct DeviceCopyKernels {

    template<class V>
    void copyBroadcast(V *d_buf, const V &v, sqaod::SizeType nElms) const;

    template<class V>
    void copyBroadcastStrided(V *d_buf, const V &v, sqaod::SizeType size,
                              sqaod::SizeType stride, sqaod::IdxType offset) const;

    DeviceCopyKernels(DeviceStream *stream = NULL);

    void assignStream(DeviceStream *stream);

private:
    cudaStream_t stream_;
};



template<class V>
void generateBitsSequence(V *d_data, int N,
                          sqaod::PackedBits xBegin, sqaod::PackedBits xEnd,
                          cudaStream_t stream);



}  // namespace sqaod_cuda

#endif
