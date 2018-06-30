#include <cub/cub.cuh>
#include "cudafuncs.h"
#include "devfuncs.cuh"
#include "DeviceKernels.h"
#include <algorithm>

#include <cuda/DeviceSegmentedSum.cuh>
#include <cuda/DeviceBatchedDot.cuh>

using namespace sqaod_cuda;

using sq::SizeType;
using sq::IdxType;
using sq::PackedBitSet;


template<class OutType, class InType>  static __global__
void scaleKernel(OutType d_y, const InType d_x, SizeType size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        d_y[gid] = (typename OutType::value_type)d_x[gid];
}

template<class OutType, class InType>
inline void scale(OutType &d_out, const InType &d_in, sq::SizeType size, cudaStream_t stream) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    scaleKernel<<<gridDim, blockDim, 0, stream>>>(d_out, d_in, size);
    DEBUG_SYNC;
}

template<class OutType, class InType>  static __global__
void scale2dKernel(OutType d_y, const InType d_x, SizeType width, SizeType height) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if ((gidx < width) && (gidy < height))
        d_y(gidx, gidy) = (typename OutType::value_type)d_x(gidx, gidy);
}

template<class OutType, class InType>
static void scale2d(OutType d_out, InType d_in, sq::SizeType width, sq::SizeType height, cudaStream_t stream) {
    dim3 blockDim(64, 2);
    for (sq::IdxType idx = 0; idx < height; idx += 65535) {
        int hSpan = std::min(height - idx, 65535);
        dim3 gridDim(divru(width, blockDim.x), divru(hSpan, blockDim.y));
        scale2dKernel <<<gridDim, blockDim, 0, stream>>>(d_out, d_in, width, height);
        d_in.addYOffset(65535);  d_out.addYOffset(65535);
        DEBUG_SYNC;
    }
}


template<class Op>  static __global__
void transform2dKernel(Op op, sq::SizeType width, sq::SizeType height, sq::IdxType offset) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y + offset;
    if ((gidx < width) && (gidy < height))
        op(gidx, gidy);
}

template<class Op>
static void transform2d(const Op &op, sq::SizeType width, sq::SizeType height, cudaStream_t stream) {
    dim3 blockDim;
    blockDim.x = std::min(roundUp(width, WARP_SIZE), 128);
    blockDim.y = 128 / blockDim.x;
    for (sq::IdxType idx = 0; idx < height; idx += 65535) {
        int hSpan = std::min(height - idx, 65535);
        dim3 gridDim(divru(width, blockDim.x), divru(hSpan, blockDim.y));
        transform2dKernel <<<gridDim, blockDim, 0, stream>>>(op, width, height, idx);
        DEBUG_SYNC;
    }
}


template<class OutType, class InType>
__global__ static
void scaleDiagonalKernel(OutType out, InType in, sq::SizeType size, sq::IdxType xOffset, sq::IdxType yOffset) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int x = gid + xOffset;
    int y = gid + yOffset;
    if (gid < size)
        out(x, y) = in[gid];
}

template<class OutType, class InType> inline void
scaleDiagonal(OutType out, InType in, sq::SizeType width, sq::SizeType height, sq::IdxType offset, cudaStream_t stream) {
    int xOffset, yOffset;
    if (0 <= offset) {
        xOffset = offset;
        yOffset = 0;
    }
    else {
        offset = -offset;
        xOffset = 0;
        yOffset = offset;
    }
    int size = std::min(width - xOffset, height - yOffset);
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    scaleDiagonalKernel<<<gridDim, blockDim, 0, stream>>>(out, in, size, xOffset, yOffset);
    DEBUG_SYNC;
}


template<class real>
void DeviceMathKernelsType<real>::scale(DeviceScalar *d_y, real alpha, const DeviceScalar &d_x, real addAssignFactor) {
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr(d_y->d_data, alpha);
        scaleKernel<<<1, 1, 0, stream_>>>(outPtr, d_x.d_data, 1);
    }
    else {
        auto outPtr = AddAssignOutPtr(d_y->d_data, addAssignFactor, alpha);
        scaleKernel<<<1, 1, 0, stream_>>>(outPtr, d_x.d_data, 1);
    }
    DEBUG_SYNC;
}

template<class real>
void DeviceMathKernelsType<real>::scale(DeviceVector *d_y, real alpha, const DeviceVector &d_x, real addAssignFactor) {
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_y->d_data, alpha);
        ::scale(outPtr, d_x.d_data, d_x.size, stream_);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_y->d_data, addAssignFactor, alpha);
        ::scale(outPtr, d_x.d_data, d_x.size, stream_);
    }
}

template<class real>
void DeviceMathKernelsType<real>::scale(DeviceMatrix *d_A, real alpha, const DeviceMatrix &d_X, real addAssignFactor) {
    auto inPtr = InPtr<real>(d_X.d_data, d_X.stride);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_A->d_data, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_X.cols, d_X.rows, stream_);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_A->d_data, addAssignFactor, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_X.cols, d_X.rows, stream_);
    }
}
    
template<class real> void DeviceMathKernelsType<real>::
scaleBroadcast(DeviceVector *d_x, real alpha, const DeviceScalar &d_c, real addAssignFactor) {
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_x->d_data, alpha);
        auto inPtr = InScalarPtr<real>(d_c.d_data);
        ::scale(outPtr, inPtr, d_x->size, stream_);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_x->d_data, addAssignFactor, alpha);
        auto inPtr = InScalarPtr<real>(d_c.d_data);
        ::scale(outPtr, inPtr, d_x->size, stream_);
    }
}

template<class real> void DeviceMathKernelsType<real>::
scaleBroadcast(DeviceMatrix *d_A, real alpha, const DeviceScalar &d_c, real addAssignFactor) {
    auto inPtr = InScalarPtr<real>(d_c.d_data);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_A->d_data, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_A->cols, d_A->rows, stream_);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_A->d_data, addAssignFactor, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_A->cols, d_A->rows, stream_);
    }
}


template<class real>
void DeviceMathKernelsType<real>::
scaleBroadcastToRows(DeviceMatrix *d_A, real alpha, const DeviceVector &d_x, real addAssignFactor) {
    auto inPtr = InRowBroadcastPtr<real>(d_x.d_data);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_A->d_data, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_A->cols, d_A->rows, stream_);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_A->d_data, addAssignFactor, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_A->cols, d_A->rows, stream_);
    }
}

template<class real>
void DeviceMathKernelsType<real>::
scaleBroadcastToColumns(DeviceMatrix *d_A, real alpha, const DeviceVector &d_x, real addAssignFactor) {
    auto inPtr = InColumnBroadcastPtr<real>(d_x.d_data);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_A->d_data, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_A->cols, d_A->rows, stream_);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_A->d_data, addAssignFactor, alpha, d_A->stride);
        ::scale2d(outPtr, inPtr, d_A->cols, d_A->rows, stream_);
    }
}

template<class real> void DeviceMathKernelsType<real>::
sum(DeviceScalar *d_dst, real alpha, const DeviceVector &d_x, real addAssignFactor) {
    size_t temp_storage_bytes;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                           d_x.d_data, d_dst->d_data, d_x.size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_dst->d_data, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               d_x.d_data, outPtr, d_x.size, stream_, CUB_DEBUG);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_dst->d_data, addAssignFactor, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               d_x.d_data, outPtr, d_x.size, stream_, CUB_DEBUG);
    }
}

template<class real>
void DeviceMathKernelsType<real>::sum(DeviceScalar *d_dst, real alpha, const DeviceMatrix &d_A, real addAssignFactor) {
    sq::SizeType size = d_A.rows * d_A.cols;
    InLinear2dPtr<real> in(d_A.d_data, d_A.stride, d_A.cols);

    size_t temp_storage_bytes;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                           in, d_dst->d_data, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_dst->d_data, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               in, outPtr, size, stream_, CUB_DEBUG);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_dst->d_data, addAssignFactor, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               in, outPtr, size, stream_, CUB_DEBUG);
    }
}

template<class real> void DeviceMathKernelsType<real>::
sumDiagonal(DeviceScalar *d_dst, real alpha, const DeviceMatrix &d_A, sq::SizeType offset, real addAssignFactor) {
    int xOffset, yOffset;
    if (0 <= offset) {
        xOffset = offset;
        yOffset = 0;
    }
    else {
        offset = -offset;
        xOffset = 0;
        yOffset = offset;
    }
    int size = std::min(d_A.cols - xOffset, d_A.rows - yOffset);

    auto inPtr = InDiagonalPtr<real>(d_A.d_data, d_A.stride, xOffset, yOffset);
    size_t temp_storage_bytes;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                           inPtr, d_A.d_data, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_dst->d_data, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               inPtr, outPtr, size, stream_, CUB_DEBUG);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_dst->d_data, addAssignFactor, alpha);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               inPtr, outPtr, size, stream_, CUB_DEBUG);
    }
}

template<class real> void DeviceMathKernelsType<real>::
sumRowwise(DeviceVector *d_x, real alpha, const DeviceMatrix &d_A) {
    auto outPtr = MulOutPtr<real>(d_x->d_data, alpha);
#if 0
    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::Sum(NULL, temp_storage_bytes,
                                    d_A.d_data, outPtr, d_A.rows,
                                    Linear(d_A.stride, 0), Linear(d_A.stride, d_A.cols),
                                    stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                    d_A.d_data, outPtr, d_A.rows,
                                    Linear(d_A.stride, 0), Linear(d_A.stride, d_A.cols),
                                    stream_, CUB_DEBUG);
    DEBUG_SYNC;
#else
    typedef DeviceBatchedSum<real, OpOutPtr<MulOutOp, real>> Sum;
    Sum &segSum = static_cast<Sum&>(*segmentedSum_);
    segSum.configure(d_A.cols, d_A.rows, true);
    segSum(d_A, outPtr);
#endif
}

template<class real> void DeviceMathKernelsType<real>::
dot(DeviceScalar *d_c, real alpha, const DeviceVector &d_x, const DeviceVector &d_y, real addAssignFactor) {

    InDotPtr<real> inPtr(d_x.d_data, d_y.d_data);
    if (addAssignFactor == 0.) {
        auto outPtr = MulOutPtr<real>(d_c->d_data, alpha);
        size_t temp_storage_bytes;
        cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                               inPtr, outPtr, d_x.size, stream_, CUB_DEBUG);
        void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);

        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               inPtr, outPtr, d_x.size, stream_, CUB_DEBUG);
    }
    else {
        auto outPtr = AddAssignOutPtr<real>(d_c->d_data, addAssignFactor, alpha);
        size_t temp_storage_bytes;
        cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                               inPtr, outPtr, d_x.size, stream_, CUB_DEBUG);
        void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);

        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               inPtr, outPtr, d_x.size, stream_, CUB_DEBUG);
    }
    DEBUG_SYNC;
}

template<class real> void DeviceMathKernelsType<real>::
dotRowwise(DeviceVector *d_z, real alpha, const DeviceMatrix &d_X, const DeviceMatrix &d_Y) {
    throwErrorIf(d_X.stride != d_Y.stride, "Strides for d_x and d_y must be same."); 

#if 0
    InDotPtr<real> inPtr(d_X.d_data, d_Y.d_data);
    auto outPtr = MulOutPtr<real>(d_z->d_data, alpha);
    InDotPtr<real> inPtr(d_X.d_data, d_y.d_data);
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
    typedef DeviceBatchedDot<real, OpOutPtr<MulOutOp, real>> Dot;
    Dot &batchedDot = static_cast<Dot&>(*segmentedDot_);
    batchedDot.configure(d_X.cols, d_X.rows, true);

    auto outPtr = MulOutPtr<real>(d_z->d_data, alpha);
    batchedDot(d_X, d_Y, outPtr);
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
transpose(DeviceMatrix *d_At, const DeviceMatrix &d_A) {
    dim3 blockDim(32, 32);
    dim3 gridDim(divru(d_A.cols, 32), divru(d_A.rows, 32));
    transposeKernel<<<gridDim, blockDim, 0, stream_>>>(d_At->d_data, d_At->stride, d_A.d_data, d_A.stride, d_A.cols, d_A.rows);
    DEBUG_SYNC;
}


template <class real>
__global__ static void
symmetrizeKernel(real *d_Asym, sq::SizeType AtStride, const real *d_A, sq::SizeType Astride, SizeType cols, SizeType rows) {

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
    
    if ((xOut < rows) && (yOut < cols)) {
        real v = d_A[xOut + AtStride * yOut];
        d_Asym[xOut + AtStride * yOut] = (v + vOut) * real(0.5);
    }
}


template<class real> void DeviceMathKernelsType<real>::
symmetrize(DeviceMatrix *d_At, const DeviceMatrix &d_A) {
    dim3 blockDim(32, 32);
    dim3 gridDim(divru(d_A.cols, 32), divru(d_A.rows, 32));
    symmetrizeKernel<<<gridDim, blockDim, 0, stream_>>>(d_At->d_data, d_At->stride, d_A.d_data, d_A.stride, d_A.cols, d_A.rows);
    DEBUG_SYNC;
}



template<class real> void DeviceMathKernelsType<real>::
min(DeviceScalar *d_min, const DeviceVector &d_x) {
    size_t temp_storage_bytes;
    cub::DeviceReduce::Min(NULL, temp_storage_bytes,
                           d_x.d_data, d_min->d_data, d_x.size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           d_x.d_data, d_min->d_data, d_x.size, stream_, CUB_DEBUG);
    DEBUG_SYNC;
}

template<class real> void DeviceMathKernelsType<real>::
min(DeviceScalar *d_min, const DeviceMatrix &d_A) {
    sq::SizeType size = d_A.rows * d_A.cols;
    InLinear2dPtr<real> in(d_A.d_data, d_A.stride, d_A.cols);

    size_t temp_storage_bytes;
    cub::DeviceReduce::Min(NULL, temp_storage_bytes,
                           in, d_min->d_data, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           in, d_min->d_data, size, stream_, CUB_DEBUG);
    DEBUG_SYNC;
}

//
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

    typedef DeviceBatchedSum<real, OpOutPtr<MulOutOp, real>> Sum;
    typedef DeviceBatchedDot<real, OpOutPtr<MulOutOp, real>> Dot;
    segmentedSum_ = new Sum(devStream_);
    segmentedDot_ = new Dot(devStream_);
}

template struct sqaod_cuda::DeviceMathKernelsType<double>;
template struct sqaod_cuda::DeviceMathKernelsType<float>;


/* DeviceCopyKernels */

template<class V> void DeviceCopyKernels::
broadcast(DeviceVectorType<V> *dst, const V &v) const {
    auto outPtr = NullOutPtr(dst->d_data);
    auto inPtr = InConstPtr<V>(v);
    scale(outPtr, inPtr, dst->size, stream_);
}

template<class V> void DeviceCopyKernels::
broadcast(DeviceMatrixType<V> *dst, const V &v) const {
    auto outPtr = NullOutPtr(dst->d_data, dst->stride);
    auto inPtr = InConstPtr<V>(v);
    scale2d(outPtr, inPtr, dst->cols, dst->rows, stream_);
}

template<class V> void DeviceCopyKernels::
broadcastToRows(DeviceMatrixType<V> *dst, const DeviceVectorType<V> &vec) const {
    auto outPtr = NullOutPtr(dst->d_data, dst->stride);
    auto inPtr = InRowBroadcastPtr<V>(vec.d_data);
    scale2d(outPtr, inPtr, dst->cols, dst->rows, stream_);
}

template<class V> void DeviceCopyKernels::
broadcastToDiagonal(DeviceMatrixType<V> *d_A, const V &v, sq::IdxType offset) const {
    auto outPtr = NullOutPtr(d_A->d_data, d_A->stride);
    auto inPtr = InConstPtr<V>(v);
    scaleDiagonal(outPtr, inPtr, d_A->cols, d_A->rows, offset, stream_);
}

template<class Vdst, class Vsrc> void DeviceCopyKernels::
cast(DeviceVectorType<Vdst> *dst, const DeviceVectorType<Vsrc> &src) {
    auto outPtr = NullOutPtr(dst->d_data);
    auto inPtr = InPtr<Vsrc>(src.d_data);
    scale(outPtr, inPtr, src.size, stream_);
}

template<class Vdst, class Vsrc> void DeviceCopyKernels::
cast(DeviceMatrixType<Vdst> *dst, const DeviceMatrixType<Vsrc> &src) {
    auto outPtr = NullOutPtr(dst->d_data, dst->stride);
    auto inPtr = InPtr<Vsrc>(src.d_data, src.stride);
    scale2d(outPtr, inPtr, src.cols, src.rows, stream_);
}

template<class V> void DeviceCopyKernels::
clearPadding(DeviceMatrixType<V> *mat) {
    int toPad = mat->stride - mat->cols;
    if (toPad != 0) {
        V *d_data = mat->d_data;
        sq::SizeType stride = mat->stride;
        sq::SizeType cols = mat->cols;
        transform2d([=]__device__(int gidx, int gidy) mutable {
            d_data[gidx + cols + gidy * stride] = V();
        }, toPad, mat->rows, stream_);
    }
}

DeviceCopyKernels::DeviceCopyKernels(DeviceStream *stream) {
    stream_ = NULL;
    if (stream != NULL)
        assignStream(stream);
}

void DeviceCopyKernels::assignStream(DeviceStream *stream) {
    stream_ = stream->getCudaStream();
}

template void DeviceCopyKernels::broadcast(DeviceVectorType<double> *dst, const double &v) const;
template void DeviceCopyKernels::broadcast(DeviceVectorType<float> *dst, const float &v) const;
template void DeviceCopyKernels::broadcast(DeviceVectorType<char> *dst, const char &v) const;

template void DeviceCopyKernels::broadcast(DeviceMatrixType<double> *dst, const double &v) const;
template void DeviceCopyKernels::broadcast(DeviceMatrixType<float> *dst, const float &v) const;
template void DeviceCopyKernels::broadcast(DeviceMatrixType<char> *dst, const char &v) const;

template void DeviceCopyKernels::broadcastToDiagonal(DeviceMatrixType<double> *, const double &, sq::IdxType) const;
template void DeviceCopyKernels::broadcastToDiagonal(DeviceMatrixType<float> *, const float &, sq::IdxType) const;
template void DeviceCopyKernels::broadcastToDiagonal(DeviceMatrixType<char> *, const char &, sq::IdxType) const;

template void DeviceCopyKernels::broadcastToRows(DeviceMatrixType<double> *, const DeviceVectorType<double> &) const;
template void DeviceCopyKernels::broadcastToRows(DeviceMatrixType<float> *, const DeviceVectorType<float> &) const;
template void DeviceCopyKernels::broadcastToRows(DeviceMatrixType<char> *, const DeviceVectorType<char> &) const;

template void DeviceCopyKernels::cast(DeviceVectorType<char> *, const DeviceVectorType<float> &);
template void DeviceCopyKernels::cast(DeviceVectorType<float> *, const DeviceVectorType<char> &);
template void DeviceCopyKernels::cast(DeviceVectorType<char> *, const DeviceVectorType<double> &);
template void DeviceCopyKernels::cast(DeviceVectorType<double> *, const DeviceVectorType<char> &);

template void DeviceCopyKernels::cast(DeviceMatrixType<char> *, const DeviceMatrixType<float> &);
template void DeviceCopyKernels::cast(DeviceMatrixType<float> *, const DeviceMatrixType<char> &);
template void DeviceCopyKernels::cast(DeviceMatrixType<char> *, const DeviceMatrixType<double> &);
template void DeviceCopyKernels::cast(DeviceMatrixType<double> *, const DeviceMatrixType<char> &);

template void DeviceCopyKernels::clearPadding(DeviceMatrixType<char> *);
template void DeviceCopyKernels::clearPadding(DeviceMatrixType<float> *);
template void DeviceCopyKernels::clearPadding(DeviceMatrixType<double> *);


template<class V>
__global__ static
void generateBitsSequenceKernel(V *d_data, sq::SizeType stride, int N,
                                SizeType nSeqs, PackedBitSet xBegin) {
    IdxType seqIdx = blockDim.y * blockIdx.x + threadIdx.y;
    if ((seqIdx < nSeqs) && (threadIdx.x < N)) {
        PackedBitSet bits = xBegin + seqIdx;
        bool bitSet = bits & (1ull << (N - 1 - threadIdx.x));
        d_data[seqIdx * stride + threadIdx.x] = bitSet ? V(1) : V(0);
    }
}

template<class V> void
sqaod_cuda::generateBitSetSequence(DeviceMatrixType<V> *d_q, PackedBitSet xBegin, PackedBitSet xEnd,
                                   cudaStream_t stream) {
    sq::SizeType N = d_q->cols;
    dim3 blockDim, gridDim;
    blockDim.x = roundUp(N, 32); /* Packed bits <= 63 bits. */
    blockDim.y = 128 / blockDim.x; /* 2 or 4, sequences per block. */
    SizeType nSeqs = sq::SizeType(xEnd - xBegin);
    gridDim.x = divru(int(xEnd - xBegin), blockDim.y);
    generateBitsSequenceKernel
            <<<gridDim, blockDim, 0, stream>>>(d_q->d_data, d_q->stride, d_q->cols, nSeqs, xBegin);
    DEBUG_SYNC;
}

template void sqaod_cuda::generateBitSetSequence(DeviceMatrixType<double> *d_q, PackedBitSet xBegin, PackedBitSet xEnd, cudaStream_t stream);
template void sqaod_cuda::generateBitSetSequence(DeviceMatrixType<float> *d_q, PackedBitSet xBegin, PackedBitSet xEnd, cudaStream_t stream);
template void sqaod_cuda::generateBitSetSequence(DeviceMatrixType<char> *d_q, PackedBitSet xBegin, PackedBitSet xEnd, cudaStream_t stream);



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
void sqaod_cuda::randomizeSpin(DeviceVectorType<V> *d_q, DeviceRandom &d_random, cudaStream_t stream) {
    dim3 blockDim(128);
    dim3 gridDim(divru(d_q->size, blockDim.x));
    sq::IdxType offset;
    sq::SizeType sizeToWrap;
    const unsigned int *d_randnum = d_random.get(d_q->size, &offset, &sizeToWrap);
    randomizeSpin2d_Kernel<<<gridDim, blockDim, 0, stream>>>(d_q->d_data, 0,
                                                             d_randnum, offset, sizeToWrap,
                                                             d_q->size, 1);
    DEBUG_SYNC;
}

template<class V>
void sqaod_cuda::randomizeSpin(DeviceMatrixType<V> *d_q, DeviceRandom &d_random, cudaStream_t stream) {
    dim3 blockDim(64, 2);
    dim3 gridDim(divru(d_q->cols, blockDim.x), divru(d_q->rows, blockDim.y));
    sq::IdxType offset;
    sq::SizeType sizeToWrap;
    const unsigned int *d_randnum = d_random.get(d_q->cols * d_q->rows, &offset, &sizeToWrap);
    randomizeSpin2d_Kernel<<<gridDim, blockDim, 0, stream>>>(d_q->d_data, d_q->stride,
                                                             d_randnum, offset, sizeToWrap,
                                                             d_q->cols, d_q->rows);
    DEBUG_SYNC;
}

template void sqaod_cuda::randomizeSpin(DeviceVectorType<double> *d_matq, DeviceRandom &d_random, cudaStream_t stream);
template void sqaod_cuda::randomizeSpin(DeviceVectorType<float> *d_matq, DeviceRandom &d_random, cudaStream_t stream);
template void sqaod_cuda::randomizeSpin(DeviceVectorType<char> *d_matq, DeviceRandom &d_random, cudaStream_t stream);
template void sqaod_cuda::randomizeSpin(DeviceMatrixType<double> *d_matq, DeviceRandom &d_random, cudaStream_t stream);
template void sqaod_cuda::randomizeSpin(DeviceMatrixType<float> *d_matq, DeviceRandom &d_random, cudaStream_t stream);
template void sqaod_cuda::randomizeSpin(DeviceMatrixType<char> *d_matq, DeviceRandom &d_random, cudaStream_t stream);
