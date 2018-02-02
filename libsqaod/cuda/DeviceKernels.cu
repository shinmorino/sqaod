#include "cudafuncs.h"
#include "DeviceKernels.h"
#include <cub/cub.cuh>

using sqaod::SizeType;
using sqaod::IdxType;
using sqaod::PackedBits;
using namespace sqaod_cuda;


/* FIXME: add __forceinline__ for device funcs/methods. */

namespace {
        
template<class real>
struct AddAssign {
    __device__ AddAssign(real &_d_value, real _mulFactor, real _alpha) : d_value(_d_value), mulFactor(_mulFactor), alpha(_alpha) { }
    __forceinline__
    __device__ real operator=(const real &v) const {
        return d_value = mulFactor * d_value + alpha * v;
    }
    real &d_value;
    real mulFactor;
    real alpha;
};

template<class real>
struct AddAssignDevPtr {
    typedef real value_type;

    AddAssignDevPtr(real *_d_data, real _mulFactor, real _alpha) : d_data(_d_data), mulFactor(_mulFactor), alpha(_alpha) { }
    typedef AddAssign<real> Ref;
    __device__ Ref operator*() const {
        return Ref(*d_data, mulFactor, alpha);
    }
    __device__ Ref operator[](SizeType idx) const {
        return Ref(d_data[idx], mulFactor, alpha);
    }

    real *d_data;
    real mulFactor;
    real alpha;
};


template<class real>
struct Mul{
    __device__ Mul(real &_d_value, real _alpha) : d_value(_d_value), alpha(_alpha) { }
    __forceinline__
    __device__ real operator=(const real &v) const {
        return d_value = alpha * v;
    }
    real &d_value;
    real alpha;
};

template<class real>
struct MulOutDevPtr {
    typedef real value_type;

    MulOutDevPtr(real *_d_data, real _alpha) : d_data(_d_data), alpha(_alpha) { }
    typedef Mul<real> Ref;
    __device__ Ref operator*() const {
        return Ref(*d_data, alpha);
    }
    __device__ Ref operator[](SizeType idx) const {
        return Ref(d_data[idx], alpha);
    }

    real *d_data;
    real alpha;
};



template<class real>
struct StridedInPtr {
    typedef real value_type;
    typedef StridedInPtr SelfType;
    __host__ __device__
    StridedInPtr(const real *_d_data, SizeType _stride, IdxType _offset) : d_data(_d_data), stride(_stride), offset(_offset) { }
    __device__ const real &operator[](SizeType idx) const {
        return d_data[offset + idx * stride];
    }
    __device__
    SelfType operator+(IdxType v) const {
        return SelfType(d_data + v, stride, offset);
    }

    const real *d_data;
    SizeType stride;
    IdxType offset;
};

}

namespace std {

template<class real>
struct iterator_traits<AddAssignDevPtr<real> > : sqaod_cuda::base_iterator_traits<real> { };
template<class real>
struct iterator_traits<MulOutDevPtr<real> > : sqaod_cuda::base_iterator_traits<real> { };
template<class real>
struct iterator_traits<StridedInPtr<real>> : sqaod_cuda::base_iterator_traits<real> { };

}




template<class OutType, class real>  static __global__
void scaleKernel(OutType d_y, real alpha, const real *d_x, SizeType size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        d_y[gid] = alpha * d_x[gid];
}

template<class real>
void DeviceMathKernelsType<real>::scale(real *d_y, real alpha, const real *d_x, SizeType size, real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    if (addAssignFactor == 0.) {
        scaleKernel <<<gridDim, blockDim, 0, stream_ >>> (d_y, alpha, d_x, size);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_y, addAssignFactor, 1.);
        scaleKernel <<<gridDim, blockDim, 0, stream_ >>> (outPtr, alpha, d_x, size);
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
void scaleBroadcastVectorKernel(OutPtrType d_A, real alpha, const real *d_x, SizeType size) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx < size) {
        SizeType pos = gidx + size * gidy;
        d_A[pos] = alpha * d_x[gidx];
    }
}

template<class real>
void DeviceMathKernelsType<real>::
scaleBroadcastVector(real *d_A, real alpha, const real *d_x, SizeType size,
                     SizeType nBatch, real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x), divru(nBatch, blockDim.y));
    if (addAssignFactor == 0.) {
        scaleBroadcastVectorKernel<<<gridDim, blockDim, 0, stream_>>>(d_A, alpha, d_x, size);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_A, addAssignFactor, real(1.));
        scaleBroadcastVectorKernel<<<gridDim, blockDim, 0, stream_>>>(outPtr, alpha, d_x, size);
    }
    DEBUG_SYNC;
}


template<class real, class OutPtrType>
static __global__
void scaleBroadcastScalarsKernel(OutPtrType d_A, real alpha, const real *d_x, SizeType size) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx < size) {
        SizeType pos = gidx + size * gidy;
        d_A[pos] = alpha * d_x[gidy];
    }
}

template<class real>
void DeviceMathKernelsType<real>::
scaleBroadcastScalars(real *d_A, real alpha, const real *d_x, SizeType size,
                     SizeType nBatch, real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x), divru(nBatch, blockDim.y));
    if (addAssignFactor == 0.) {
        scaleBroadcastScalarsKernel
                <<<gridDim, blockDim, 0, stream_>>>(d_A, alpha, d_x, size);
    }
    else {
        AddAssignDevPtr<real> outPtr(d_A, addAssignFactor, real(1.));
        scaleBroadcastScalarsKernel
                <<<gridDim, blockDim, 0, stream_>>>(outPtr, alpha, d_x, size);
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



template<class real> void DeviceMathKernelsType<real>::
sumGather(real *d_sum, real alpha, const real *d_x, SizeType size, SizeType stride, int offset) {
    size_t temp_storage_bytes;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes,
                           d_x, d_sum, size, stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    StridedInPtr<real> inPtr(d_x, stride, offset);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                           inPtr, d_sum, size, stream_, CUB_DEBUG);
}


namespace {
/* Functors for offsets */

struct Linear {
    Linear(IdxType _a, IdxType _b) : a(_a), b(_b) { }
    __device__
    IdxType operator[](IdxType idx) const { return a * idx + b; }
    IdxType a, b;
};

}

template<class real> void DeviceMathKernelsType<real>::
sumBatched(real *d_sum, real alpha, const real *d_A, SizeType size, SizeType nBatch) {
    MulOutDevPtr<real> outPtr(d_sum, alpha);

    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::Sum(NULL, temp_storage_bytes,
                                    d_A, outPtr, nBatch,
                                    Linear(size, 0), Linear(size, size),
                                    stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                    d_A, outPtr, nBatch,
                                    Linear(size, 0), Linear(size, size),
                                    stream_, CUB_DEBUG);
    DEBUG_SYNC;
}


namespace {

template<class real>
struct InDotPtr {
    typedef InDotPtr<real> SelfType;
    
    __host__ __device__
    InDotPtr(const real *_d_x, const real *_d_y) : d_x(_d_x), d_y(_d_y) { }
    __device__
    real operator[](IdxType idx) const {
        return d_x[idx] * d_y[idx];
    }
    __device__
    SelfType operator+(IdxType idx) const {
        return SelfType(&d_x[idx], &d_y[idx]);
    }
    
    const real *d_x, *d_y;
};

}

namespace std {

template<class real>
struct iterator_traits<InDotPtr<real>> : sqaod_cuda::base_iterator_traits<real> { };

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
dotBatched(real *d_z, real alpha, const real *d_x, const real *d_y, SizeType size,
           SizeType nBatch) {
    
    InDotPtr<real> inPtr(d_x, d_y);
    MulOutDevPtr<real> outPtr(d_z, alpha);
    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::Sum(NULL, temp_storage_bytes,
                                    inPtr, outPtr, nBatch,
                                    Linear(size, 0), Linear(size, size),
                                    stream_, CUB_DEBUG);
    void *d_temp_storage = devStream_->allocate(temp_storage_bytes, __func__);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                    inPtr, outPtr, nBatch,
                                    Linear(size, 0), Linear(size, size),
                                    stream_, CUB_DEBUG);
    DEBUG_SYNC;
}

template <class real>
__global__ static void
transposeKernel(real *d_At, const real *d_A, SizeType cols, SizeType rows) {

    int inTileLeft = blockDim.x * blockIdx.x;
    int inTileTop = blockDim.y * blockIdx.y;
    
    int xIn = inTileLeft + threadIdx.x;
    int yIn = inTileTop + threadIdx.y;

    real vIn = (xIn < cols) && (yIn < rows) ? d_A[xIn + cols * yIn] : real();

    __shared__ real tile[32][33];
    tile[threadIdx.y][threadIdx.x] = vIn;
	__syncthreads();

    int xOut = inTileTop + threadIdx.x;
    int yOut = inTileLeft + threadIdx.y;
    real vOut = tile[threadIdx.x][threadIdx.y];
    
    if ((xOut < rows) && (yOut < cols))
        d_At[xOut + rows * yOut] = vOut;
}


template<class real> void DeviceMathKernelsType<real>::
transpose(real *d_At, const real *d_A, SizeType rows, SizeType cols) {
    dim3 blockDim(32, 32);
    dim3 gridDim(divru(cols, 32u), divru(rows, 32u));
    transposeKernel<<<gridDim, blockDim, 0, stream_>>>(d_At, d_A, cols, rows);
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


template<> void DeviceMathKernelsType<double>::
gemv(cublasOperation_t op, int M, int N,
     const double *d_alpha, const double *d_A, const double *d_x,
     const double *d_beta, double *d_y) {
    throwOnError(cublasDgemv(devStream_->getCublasHandle(), op, M, N, d_alpha, d_A, M, d_x, 1, d_beta, d_y, 1));
}

template<> void DeviceMathKernelsType<float>::
gemv(cublasOperation_t op, int M, int N,
     const float *d_alpha, const float *d_A, const float *d_x,
     const float *d_beta, float *d_y) {
    throwOnError(cublasSgemv(devStream_->getCublasHandle(), op, M, N, d_alpha, d_A, M, d_x, 1, d_beta, d_y, 1));
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
copyBroadcast(V *d_buf, const V &v, sqaod::SizeType size) const {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    copyBroadcastKernel<<<gridDim, blockDim, 0, stream_>>>(d_buf, v, size);
    DEBUG_SYNC;
}


template<class V>
__global__ static
void copyBroadcastStridedKernel(V *d_buf, const V v, SizeType size, SizeType stride, IdxType offset) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        IdxType pos = gid * stride + offset;
        d_buf[pos] = v;
    }
}


template<class V> void DeviceCopyKernels::
copyBroadcastStrided(V *d_buf, const V &v, SizeType size, SizeType stride, IdxType offset) const {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    copyBroadcastStridedKernel<<<gridDim, blockDim, 0, stream_>>>(d_buf, v, size, stride, offset);
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


template void DeviceCopyKernels::copyBroadcastStrided(double *, const double &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(float *, const float &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(char *, const char &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(unsigned char *, const unsigned char &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(short *, const short &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(unsigned short *, const unsigned short &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(int *, const int &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(unsigned int *, const unsigned int &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(long *, const long &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(unsigned long *, const unsigned long &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(long long *, const long long &, SizeType, SizeType, IdxType) const;
template void DeviceCopyKernels::copyBroadcastStrided(unsigned long long *, const unsigned long long &, SizeType, SizeType, IdxType) const;

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


    

template<class V>
__global__ static
void generateBitsSequenceKernel(V *d_data, int N,
                                SizeType nSeqs, PackedBits xBegin) {
    IdxType seqIdx = blockDim.y * blockIdx.x + threadIdx.y;
    if ((seqIdx < nSeqs) && (threadIdx.x < N)) {
        PackedBits bits = xBegin + seqIdx;
        bool bitSet = bits & (1ull << (N - 1 - threadIdx.x));
        d_data[seqIdx * N + threadIdx.x] = bitSet ? V(1) : V(0);
    }
}


template<class V> void
sqaod_cuda::generateBitsSequence(V *d_data, int N, PackedBits xBegin, PackedBits xEnd,
                                 cudaStream_t stream) {
    dim3 blockDim, gridDim;
    blockDim.x = roundUp(N, 32); /* Packed bits <= 63 bits. */
    blockDim.y = 128 / blockDim.x; /* 2 or 4, sequences per block. */
    SizeType nSeqs = SizeType(xEnd - xBegin);
    gridDim.x = (unsigned int)divru((unsigned int)(xEnd - xBegin), blockDim.y);
    generateBitsSequenceKernel
            <<<gridDim, blockDim, 0, stream>>>(d_data, N, nSeqs, xBegin);
    DEBUG_SYNC;
}
