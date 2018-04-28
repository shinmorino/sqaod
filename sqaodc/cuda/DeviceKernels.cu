#include <cub/cub.cuh>
#include "cub_iterator.cuh"
#include "cudafuncs.h"
#include "DeviceKernels.h"

#include <cuda/DeviceSegmentedSum.cuh>

using namespace sqaod_cuda;

using sq::SizeType;
using sq::IdxType;
using sq::PackedBitSet;


template<class OutType, class real>  static __global__
void scale2dKernel(OutType d_y,
                   real alpha, const real *d_x, SizeType stride, SizeType rows, SizeType cols) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((gidx < cols) && (gidy < rows))
        d_y[gidx + gidy * stride] = alpha * d_x[gidx + gidy * stride];
}

template<class real>
void DeviceMathKernelsType<real>::scale(real *d_y, real alpha, const real *d_x, SizeType size, real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    if (addAssignFactor == 0.) {
        scale2dKernel<<<gridDim, blockDim, 0, stream_>>>(d_y, alpha, d_x, 0, 1, size);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_y, addAssignFactor, 1.);
        scale2dKernel<<<gridDim, blockDim, 0, stream_ >>> (outPtr, alpha, d_x, 0, 1, size);
    }
    DEBUG_SYNC;
}

template<class real>
void DeviceMathKernelsType<real>::scale2d(real *d_y, sq::SizeType yStride,
                                          real alpha, const real *d_x, sq::SizeType xStride,
                                          sq::SizeType rows, sq::SizeType cols, real addAssignFactor) {
    throwErrorIf(yStride != xStride, "Strides for x and y are not same.");
    dim3 blockDim(64, 2);
    dim3 gridDim(divru(cols, blockDim.x), divru(rows, blockDim.y));
    if (addAssignFactor == 0.) {
        scale2dKernel<<<gridDim, blockDim, 0, stream_ >>> (d_y, alpha, d_x, xStride, rows, cols);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_y, addAssignFactor, 1.);
        scale2dKernel<<<gridDim, blockDim, 0, stream_ >>> (outPtr, alpha, d_x, xStride, rows, cols);
    }
    DEBUG_SYNC;
}

template<class real, class OutType>
static __global__
void scaleBroadcastKernel(OutType d_y, real alpha, const real *d_c, SizeType size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        d_y[gid] = alpha * (*d_c);
}

template<class real> void DeviceMathKernelsType<real>::
scaleBroadcast(real *d_y, real alpha, const real *d_c, SizeType size,
               real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    if (addAssignFactor == 0.) {
        scaleBroadcastKernel<<<gridDim, blockDim, 0, stream_>>>
                (d_y, alpha, d_c, size);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_y, addAssignFactor, real(1.));
        scaleBroadcastKernel<real><<<gridDim, blockDim, 0, stream_>>>(outPtr, alpha, d_c, size);
    }
    DEBUG_SYNC;
}

template<class real, class OutPtrType>  static __global__
void scaleBroadcastVectorKernel(OutPtrType d_A, sq::IdxType Astride,
                                real alpha, const real *d_x, SizeType size) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx < size) {
        SizeType pos = gidx + Astride * gidy;
        d_A[pos] = alpha * d_x[gidx];
    }
}

template<class real>
void DeviceMathKernelsType<real>::
scaleBroadcastVector(real *d_A, sq::SizeType Astride,
                     real alpha, const real *d_x, SizeType size,
                     SizeType nBatch, real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x), divru(nBatch, blockDim.y));
    if (addAssignFactor == 0.) {
        scaleBroadcastVectorKernel<<<gridDim, blockDim, 0, stream_>>>
                (d_A, Astride, alpha, d_x, size);
        throwOnError(cudaGetLastError());
    }
    else {
        AddAssignDevPtr<real> outPtr(d_A, addAssignFactor, real(1.));
        scaleBroadcastVectorKernel<<<gridDim, blockDim, 0, stream_>>>
                (outPtr, Astride, alpha, d_x, size);
        throwOnError(cudaGetLastError());
    }
    DEBUG_SYNC;
}


template<class real, class OutPtrType>
static __global__
void scaleBroadcastScalarsKernel(OutPtrType d_A, sq::IdxType stride,
                                 real alpha, const real *d_x, SizeType size) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx < size) {
        SizeType pos = gidx + stride * gidy;
        d_A[pos] = alpha * d_x[gidy];
    }
}

template<class real>
void DeviceMathKernelsType<real>::
scaleBroadcastScalars(real *d_A, sq::SizeType Astride, real alpha, const real *d_x, SizeType size,
                     SizeType nBatch, real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x), divru(nBatch, blockDim.y));
    if (addAssignFactor == 0.) {
        scaleBroadcastScalarsKernel
                <<<gridDim, blockDim, 0, stream_>>>(d_A, Astride, alpha, d_x, size);
        throwOnError(cudaGetLastError());
    }
    else {
        AddAssignDevPtr<real> outPtr(d_A, addAssignFactor, real(1.));
        scaleBroadcastScalarsKernel
                <<<gridDim, blockDim, 0, stream_>>>(outPtr, Astride, alpha, d_x, size);
        throwOnError(cudaGetLastError());
    }
    DEBUG_SYNC;
}


template<class real> void DeviceMathKernelsType<real>::
sum(real *d_sum, real alpha, const real *d_x, SizeType size, real addAssignFactor) {
    size_t temp_storage_bytes;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                           d_x, d_sum, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    if (addAssignFactor == 0.) {
        MulOutDevPtr<real> outPtr(d_sum, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               d_x, outPtr, size, stream_, CUB_DEBUG);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_sum, addAssignFactor, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               d_x, outPtr, size, stream_, CUB_DEBUG);
    }
}


template<class real>
void DeviceMathKernelsType<real>::sum2d(real *d_sum,
                                        real alpha, const real *d_values, sq::SizeType stride,
                                        sq::SizeType rows, sq::SizeType cols, real addAssignFactor) {
    sq::SizeType size = rows * cols;
    In2dPtr<real> in(d_values, stride, cols);

    size_t temp_storage_bytes;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                           in, d_sum, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           in, d_sum, size, stream_, CUB_DEBUG);
    DEBUG_SYNC;


}

template<class real> void DeviceMathKernelsType<real>::
sumWithInterval(real *d_sum, real alpha, const real *d_x, SizeType interval, int offset, SizeType size) {
    size_t temp_storage_bytes;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                           d_x, d_sum, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    InPtrWithInterval<real> inPtr(d_x, interval, offset);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           inPtr, d_sum, size, stream_, CUB_DEBUG);
}


template<class real> void DeviceMathKernelsType<real>::
sumBatched(real *d_sum, real alpha, const real *d_A, sq::SizeType Astride,
           sq::SizeType size, sq::SizeType nBatch) {
    MulOutDevPtr<real> outPtr(d_sum, alpha);
#if 0
    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::Sum(NULL, temp_storage_bytes,
                                    d_A, outPtr, nBatch,
                                    Linear(Astride, 0), Linear(Astride, size),
                                    stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                    d_A, outPtr, nBatch,
                                    Linear(Astride, 0), Linear(Astride, size),
                                    stream_, CUB_DEBUG);
    DEBUG_SYNC;
#else
    typedef DeviceSegmentedSumTypeImpl<real, const real*, MulOutDevPtr<real>, Linear> Sum;
    Sum &segSum = static_cast<Sum&>(*segmentedSum_);
    segSum.configure(size, nBatch, true);
    segSum(d_A, outPtr, Linear(Astride, 0));
#endif
}



template<class real> void DeviceMathKernelsType<real>::
dot(real *d_c, real alpha, const real *d_x, const real *d_y, SizeType size,
    real addAssignFactor) {

    InDotPtr<real> inPtr(d_x, d_y);
    if (addAssignFactor == 0.) {
        MulOutDevPtr<real> outPtr(d_c, alpha);
        size_t temp_storage_bytes;
        cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                               inPtr, outPtr, size, stream_, CUB_DEBUG);
        void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);

        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               inPtr, outPtr, size, stream_, CUB_DEBUG);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_c, addAssignFactor, alpha);
        size_t temp_storage_bytes;
        cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                               inPtr, outPtr, size, stream_, CUB_DEBUG);
        void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);

        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               inPtr, outPtr, size, stream_, CUB_DEBUG);
    }
    DEBUG_SYNC;
}


template<class real> void DeviceMathKernelsType<real>::
dotBatched(real *d_z, real alpha, const real *d_x, sq::SizeType xStride, const real *d_y, sq::SizeType yStride,
           SizeType size, SizeType nBatch) {
    
    InDotPtr<real> inPtr(d_x, d_y);
    MulOutDevPtr<real> outPtr(d_z, alpha);
    throwErrorIf(xStride != yStride, "Strides for d_x and d_y must be same."); 
#if 0
    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::Sum(NULL, temp_storage_bytes,
                                    inPtr, outPtr, nBatch,
                                    Linear(xStride, 0), Linear(xStride, size),
                                    stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                    inPtr, outPtr, nBatch,
                                    Linear(xStride, 0), Linear(xStride, size),
                                    stream_, CUB_DEBUG);
    DEBUG_SYNC;
#else
    typedef DeviceSegmentedSumTypeImpl<real, InDotPtr<real>, MulOutDevPtr<real>, Linear> Dot;
    Dot &segDot = static_cast<Dot&>(*segmentedDot_);
    segDot.configure(size, nBatch, true);
    segDot(inPtr, outPtr, Linear(xStride, 0));
#endif
}

template <class real>
__global__ static void
transposeKernel(real *d_At, sq::SizeType AtStride, const real *d_A, sq::SizeType Astride, SizeType cols, SizeType rows) {

    int inTileLeft = blockDim.x * blockIdx.x;
    int inTileTop = blockDim.y * blockIdx.y;
    
    int xIn = inTileLeft + threadIdx.x;
    int yIn = inTileTop + threadIdx.y;

    real vIn = (xIn < cols) && (yIn < rows) ? d_A[xIn + Astride * yIn] : real();

    __shared__ real tile[32][33];
    tile[threadIdx.y][threadIdx.x] = vIn;
	__syncthreads();

    int xOut = inTileTop + threadIdx.x;
    int yOut = inTileLeft + threadIdx.y;
    real vOut = tile[threadIdx.x][threadIdx.y];
    
    if ((xOut < rows) && (yOut < cols))
        d_At[xOut + AtStride * yOut] = vOut;
}


template<class real> void DeviceMathKernelsType<real>::
transpose2d(real *d_At, sq::SizeType AtStride, const real *d_A, sq::SizeType Astride, SizeType rows, SizeType cols) {
    dim3 blockDim(32, 32);
    dim3 gridDim(divru(cols, 32), divru(rows, 32));
    transposeKernel<<<gridDim, blockDim, 0, stream_>>>(d_At, AtStride, d_A, Astride, cols, rows);
    DEBUG_SYNC;
}


template<class real> void DeviceMathKernelsType<real>::
min(real *d_min, const real *d_values, SizeType size) {
    size_t temp_storage_bytes;
    cub::DeviceReduce::Min(NULL, temp_storage_bytes,
                           d_values, d_min, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           d_values, d_min, size, stream_, CUB_DEBUG);
    DEBUG_SYNC;
}

template<class real> void DeviceMathKernelsType<real>::
min2d(real *d_min,
      const real *d_values, sq::SizeType stride, sq::SizeType rows, sq::SizeType cols) {
    sq::SizeType size = rows * cols;
    In2dPtr<real> in(d_values, stride, cols);

    size_t temp_storage_bytes;
    cub::DeviceReduce::Min(NULL, temp_storage_bytes,
                           in, d_min, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           in, d_min, size, stream_, CUB_DEBUG);
    DEBUG_SYNC;
}


template<> void DeviceMathKernelsType<double>::
gemv(cublasOperation_t op, int M, int N,
     const double *d_alpha, const double *d_A, sq::SizeType Astride, const double *d_x,
     const double *d_beta, double *d_y) {
    throwOnError(cublasDgemv(devStream_->getCublasHandle(), op, M, N, d_alpha, d_A, Astride, d_x, 1, d_beta, d_y, 1));
}

template<> void DeviceMathKernelsType<float>::
gemv(cublasOperation_t op, int M, int N,
     const float *d_alpha, const float *d_A, sq::SizeType Astride, const float *d_x,
     const float *d_beta, float *d_y) {
    throwOnError(cublasSgemv(devStream_->getCublasHandle(), op, M, N, d_alpha, d_A, Astride, d_x, 1, d_beta, d_y, 1));
}

template<> void DeviceMathKernelsType<double>::
gemm(cublasOperation_t opA, cublasOperation_t opB, int M, int N, int K,
     const double *d_alpha, const double *d_A, int lda, const double *d_B, int ldb,
     const double *d_beta, double *d_C, int ldc) {
    throwOnError(cublasDgemm(devStream_->getCublasHandle(), opA, opB, M, N, K, d_alpha, d_A, lda, d_B, ldb, d_beta, d_C, ldc));
}

template<> void DeviceMathKernelsType<float>::
gemm(cublasOperation_t opA, cublasOperation_t opB, int M, int N, int K,
     const float *d_alpha, const float *d_A, int lda, const float *d_B, int ldb,
     const float *d_beta, float *d_C, int ldc) {
    throwOnError(cublasSgemm(devStream_->getCublasHandle(), opA, opB, M, N, K, d_alpha, d_A, lda, d_B, ldb, d_beta, d_C, ldc));
}

template<class real> DeviceMathKernelsType<real>::
DeviceMathKernelsType(DeviceStream *devStream) {
    devStream_ = devStream;
    stream_ = NULL;
    if (devStream != NULL)
        assignStream(devStream);
}

template<class real> void DeviceMathKernelsType<real>::
assignStream(DeviceStream *devStream) {
    devStream_ = devStream;
    stream_ = NULL;
    if (devStream_ != NULL)
        stream_ = devStream_->getCudaStream();

    typedef DeviceSegmentedSumTypeImpl<real, const real *, MulOutDevPtr<real>, Linear> Sum;
    typedef DeviceSegmentedSumTypeImpl<real, InDotPtr<real>, MulOutDevPtr<real>, Linear> Dot;
    segmentedSum_ = new Sum(devStream_);
    segmentedDot_ = new Dot(devStream_);
}

template struct sqaod_cuda::DeviceMathKernelsType<double>;
template struct sqaod_cuda::DeviceMathKernelsType<float>;


/* DeviceCopyKernels */

template<class V>
__global__ static
void copyBroadcastKernel(V *d_buf, const V v, SizeType size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        d_buf[gid] = v;
}


template<class V> void DeviceCopyKernels::
copyBroadcast(V *d_buf, const V &v, sq::SizeType size) const {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    copyBroadcastKernel<<<gridDim, blockDim, 0, stream_>>>(d_buf, v, size);
    DEBUG_SYNC;
}


template<class V>
__global__ static
void copyBroadcastWithIntervalKernel(V *d_buf, SizeType stride, IdxType offset, const V v, SizeType size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        IdxType pos = gid * stride + offset;
        d_buf[pos] = v;
    }
}

template<class V> void DeviceCopyKernels::
copyBroadcastWithInterval(V *d_buf, SizeType stride, IdxType offset, const V &v, SizeType size) const {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    copyBroadcastWithIntervalKernel<<<gridDim, blockDim, 0, stream_>>>(d_buf, stride, offset, v, size);
    DEBUG_SYNC;
}

template<class V>
__global__ static
void copyBroadcast2dKernel(V *d_buf, sq::SizeType stride, const V v, sq::SizeType rows, sq::SizeType cols) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((gidx < cols) && (gidy < rows))
        d_buf[gidx + gidy * stride] = v;
}


template<class V> void DeviceCopyKernels::
copyBroadcast2d(V *d_buf, sq::SizeType stride, const V &v, sq::SizeType rows, sq::SizeType cols) const {
    dim3 blockDim(64, 2);
    dim3 gridDim(divru(cols, blockDim.x), divru(rows, blockDim.y));
    copyBroadcast2dKernel<<<gridDim, blockDim, 0, stream_>>>(d_buf, stride, v, rows, cols);
    DEBUG_SYNC;
}


template<class V>
__global__ static void
copyBroadcastVectorKernel(V *d_dst, sq::SizeType stride, const V *d_src, sq::SizeType size, sq::SizeType nBatch) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((gidx < size) && (gidy < nBatch)) {
        d_dst[gidx + stride * gidy] = d_src[gidx];
    }
}

template<class V> inline void DeviceCopyKernels::
copyBroadcastVector(V *dst, sq::SizeType stride, const V *vec, sq::SizeType size, sq::SizeType nBatch) const {
    dim3 blockDim(64, 2);
    dim3 gridDim(divru(size, blockDim.x), divru(nBatch, blockDim.y));
    copyBroadcastVectorKernel << <gridDim, blockDim >> > (dst, stride, vec, size, nBatch);
}


template<class D, class S>
__global__ static void
cast2dKernel(D *d_dst, sq::SizeType dstStride, const S *d_src, sq::SizeType srcStride,
             sq::SizeType cols, sq::SizeType rows) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((gidx < cols) && (gidy < rows)) {
        d_dst[gidx + dstStride * gidy] = (D)d_src[gidx + srcStride * gidy];
    }
}


template<class Vdst, class Vsrc> void DeviceCopyKernels::
cast(Vdst *dst, const Vsrc *src, sq::SizeType size) {
    cast2d(dst, 0, src, 0, 1, size);
}

template<class Vdst, class Vsrc> void DeviceCopyKernels::
cast2d(Vdst *dst, sq::SizeType dstStride, const Vsrc *src, sq::SizeType srcStride,
       sq::SizeType rows, sq::SizeType cols) {
    dim3 blockDim(128);
    dim3 gridDim(divru(cols, blockDim.x), divru(rows, blockDim.y));
    cast2dKernel<<<gridDim, blockDim >>> (dst, dstStride, src, srcStride, rows, cols);
    DEBUG_SYNC;
}


DeviceCopyKernels::DeviceCopyKernels(DeviceStream *stream) {
    stream_ = NULL;
    if (stream != NULL)
        assignStream(stream);
}


void DeviceCopyKernels::assignStream(DeviceStream *stream) {
    stream_ = stream->getCudaStream();
}


template void DeviceCopyKernels::copyBroadcastWithInterval(double *, SizeType, IdxType, const double &, SizeType) const;

template void DeviceCopyKernels::copyBroadcastWithInterval(float *, SizeType, IdxType, const float &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(char *, SizeType, IdxType, const char &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(unsigned char *, SizeType, IdxType, const unsigned char &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(short *, SizeType, IdxType, const short &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(unsigned short *, SizeType, IdxType, const unsigned short &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(int *, SizeType, IdxType, const int &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(unsigned int *, SizeType, IdxType, const unsigned int &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(long *, SizeType, IdxType, const long &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(unsigned long *, SizeType, IdxType, const unsigned long &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(long long *, SizeType, IdxType, const long long &, SizeType) const;
template void DeviceCopyKernels::copyBroadcastWithInterval(unsigned long long *, SizeType, IdxType, const unsigned long long &, SizeType) const;

template void DeviceCopyKernels::copyBroadcast(double *, const double &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(float *, const float &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(char *, const char &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(unsigned char *, const unsigned char &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(short *, const short &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(unsigned short *, const unsigned short &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(int *, const int &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(unsigned int *, const unsigned int &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(long *, const long &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(unsigned long *, const unsigned long &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(long long *, const long long &, SizeType) const;
template void DeviceCopyKernels::copyBroadcast(unsigned long long *, const unsigned long long &, SizeType) const;

template void DeviceCopyKernels::copyBroadcast2d(double *, sq::SizeType, const double &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(float *, sq::SizeType, const float &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(char *, sq::SizeType, const char &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(unsigned char *, sq::SizeType, const unsigned char &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(short *, sq::SizeType, const short &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(unsigned short *, sq::SizeType, const unsigned short &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(int *, sq::SizeType, const int &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(unsigned int *, sq::SizeType, const unsigned int &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(long *, sq::SizeType, const long &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(unsigned long *, sq::SizeType, const unsigned long &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(long long *, sq::SizeType, const long long &, SizeType, SizeType) const;
template void DeviceCopyKernels::copyBroadcast2d(unsigned long long *, sq::SizeType, const unsigned long long &, SizeType, SizeType) const;

template void DeviceCopyKernels::cast(char *dst, const float *src, sq::SizeType size);
template void DeviceCopyKernels::cast(char *dst, const double *src, sq::SizeType size);
template void DeviceCopyKernels::cast(float *dst, const char *src, sq::SizeType size);
template void DeviceCopyKernels::cast(double *dst, const char *src, sq::SizeType size);

template void DeviceCopyKernels::copyBroadcastVector(char *dst, sq::SizeType stride, const char *vec, sq::SizeType size, sq::SizeType nBatch) const;
template void DeviceCopyKernels::copyBroadcastVector(float *dst, sq::SizeType stride, const float *vec, sq::SizeType size, sq::SizeType nBatch) const;
template void DeviceCopyKernels::copyBroadcastVector(double *dst, sq::SizeType stride, const double *vec, sq::SizeType size, sq::SizeType nBatch) const;


template<class V>
__global__ static
void generateBitsSequenceKernel(V *d_data, int N,
                                SizeType nSeqs, PackedBitSet xBegin) {
    IdxType seqIdx = blockDim.y * blockIdx.x + threadIdx.y;
    if ((seqIdx < nSeqs) && (threadIdx.x < N)) {
        PackedBitSet bits = xBegin + seqIdx;
        bool bitSet = bits & (1ull << (N - 1 - threadIdx.x));
        d_data[seqIdx * N + threadIdx.x] = bitSet ? V(1) : V(0);
    }
}


template<class V> void
sqaod_cuda::generateBitsSequence(V *d_data, int N, PackedBitSet xBegin, PackedBitSet xEnd,
                                 cudaStream_t stream) {
    dim3 blockDim, gridDim;
    blockDim.x = roundUp(N, 32); /* Packed bits <= 63 bits. */
    blockDim.y = 128 / blockDim.x; /* 2 or 4, sequences per block. */
    SizeType nSeqs = xEnd - xBegin;
    gridDim.x = divru(xEnd - xBegin, blockDim.y);
    generateBitsSequenceKernel
            <<<gridDim, blockDim, 0, stream>>>(d_data, N, nSeqs, xBegin);
    DEBUG_SYNC;
}

template<class V>
__global__ static void
randomizeSpin2d_Kernel(V *d_buffer, sq::SizeType stride,
                       const unsigned int *d_random, sq::IdxType offset, sq::SizeType sizeToWrap,
                       sq::SizeType width, sq::SizeType height) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((gidx < width) && (gidy < height))
        d_buffer[gidx + gidy * stride] =
                (d_random[(gidx + gidy * width + offset) % sizeToWrap] & 1) ? V(1) : V(-1);
}


template<class V>
void sqaod_cuda::randomizeSpin(V *d_q, DeviceRandom &d_random, sq::SizeType size,
                               cudaStream_t stream) {
    randomizeSpin2d(d_q, 0, d_random, 1, size, stream);
}

template<class V>
void sqaod_cuda::randomizeSpin2d(V *d_q, sq::SizeType stride, DeviceRandom &d_random,
                                 sq::SizeType rows, sq::SizeType cols,
                                 cudaStream_t stream) {
    dim3 blockDim(128);
    dim3 gridDim(divru(cols, blockDim.x), divru(rows, blockDim.y));
    sq::IdxType offset;
    sq::SizeType sizeToWrap;
    const unsigned int *d_randnum = d_random.get(rows * cols, &offset, &sizeToWrap);
    randomizeSpin2d_Kernel<<<gridDim, blockDim, 0, stream>>>(d_q, stride,
                                                             d_randnum, offset, sizeToWrap,
                                                             rows, cols);
    DEBUG_SYNC;
}

template void sqaod_cuda::randomizeSpin(float *d_matq, DeviceRandom &d_random, sq::SizeType size, cudaStream_t stream);
template void sqaod_cuda::randomizeSpin(double *d_matq, DeviceRandom &d_random, sq::SizeType size, cudaStream_t stream);
template void sqaod_cuda::randomizeSpin(char *d_matq, DeviceRandom &d_random, sq::SizeType size, cudaStream_t stream);
