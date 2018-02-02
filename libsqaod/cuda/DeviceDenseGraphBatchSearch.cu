#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include "DeviceDenseGraphBatchSearch.h"
#include "Device.h"


using namespace sqaod_cuda;
namespace sq = sqaod;


template<class real>
DeviceDenseGraphBatchSearch<real>::DeviceDenseGraphBatchSearch() {
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::assignDevice(Device &device) {
    devStream_ = device.defaultStream();
    dgFuncs_.assignDevice(device, devStream_);
    devCopy_.assignDevice(device, devStream_);
    devAlloc_ = device.objectAllocator();
}

template<class real>
void DeviceDenseGraphBatchSearch<real>::deallocate() {
    devAlloc_->deallocate(d_bitsMat_);
    devAlloc_->deallocate(d_Ebatch_);

    HostObjectAllocator halloc;
    halloc.deallocate(h_nXMins_);
    halloc.deallocate(h_Emin_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::setProblem(const HostMatrix &W, sq::SizeType tileSize) {
    devCopy_(&d_W_, W);
    tileSize_ = tileSize;
    devAlloc_->allocate(&d_bitsMat_, tileSize, W.rows);
    devAlloc_->allocate(&d_Ebatch_, tileSize);
    devAlloc_->allocate(&d_xMins_, tileSize * 2);

    HostObjectAllocator halloc;
    halloc.allocate(&h_nXMins_);
    halloc.allocate(&h_Emin_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::calculate_E(sq::PackedBits xBegin, sq::PackedBits xEnd) {
    xBegin_ = xBegin;
    sq::SizeType nBatch = sq::SizeType(xEnd - xBegin);
    abortIf(tileSize_ < nBatch,
            "nBatch is too large, tileSize=%d, nBatch=%d", int(tileSize_), int(nBatch));
    int N = d_W_.rows;
    generateBitsSequence(d_bitsMat_.d_data, N, xBegin, xEnd);
    dgFuncs_.calculate_E(&d_Ebatch_, d_W_, d_bitsMat_);
    dgFuncs_.devMath.min(&h_Emin_, d_Ebatch_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::partition_xMins(bool append) {
    assert(d_Ebatch_.size == tileSize_);
    if (!append) {
        /* overwrite */
        d_xMins_.size = 0;
        select(d_xMins_.d_data, h_nXMins_.d_data,
               xBegin_, *h_Emin_.d_data, d_Ebatch_.d_data, tileSize_);
        synchronize();
        d_xMins_.size = *h_nXMins_.d_data; /* sync field */
    }
    else if (d_xMins_.size < tileSize_) {
        /* append */
        select(&d_xMins_.d_data[d_xMins_.size], h_nXMins_.d_data,
               xBegin_, *h_Emin_.d_data, d_Ebatch_.d_data, tileSize_);
        synchronize();
        d_xMins_.size += *h_nXMins_.d_data; /* sync field */
    }
}

template<class real>
void DeviceDenseGraphBatchSearch<real>::synchronize() {
    devStream_->synchronize();
}


template<class real>
__global__ static
void generateBitsSequenceKernel(real *d_data, int N,
                                sq::SizeType nSeqs, sq::PackedBits xBegin) {
    sq::IdxType seqIdx = blockDim.y * blockIdx.x + threadIdx.y;
    if ((seqIdx < nSeqs) && (threadIdx.x < N)) {
        sq::PackedBits bits = xBegin + seqIdx;
        bool bitSet = bits & (1ull << (N - 1 - threadIdx.x));
        d_data[seqIdx * N + threadIdx.x] = bitSet ? real(1) : real(0);
    }
}


template<class real> void DeviceDenseGraphBatchSearch<real>::
generateBitsSequence(real *d_data, int N,
                     sq::PackedBits xBegin, sq::PackedBits xEnd) {
    dim3 blockDim, gridDim;
    blockDim.x = roundUp(N, 32); /* Packed bits <= 63 bits. */
    blockDim.y = 128 / blockDim.x; /* 2 or 4, sequences per block. */
    sq::SizeType nSeqs = sq::SizeType(xEnd - xBegin);
    gridDim.x = (unsigned int)divru((unsigned int)(xEnd - xBegin), blockDim.y);
    generateBitsSequenceKernel
            <<<gridDim, blockDim, 0, devStream_->getCudaStream()>>>(d_data, N, nSeqs, xBegin);
    DEBUG_SYNC;
}


namespace {

struct SelectInputIterator {
    __device__ __forceinline__
    SelectInputIterator(unsigned int _offset) : offset(_offset) { }
    __host__
    SelectInputIterator() : offset(0) { }

    __device__ __forceinline__
    sq::PackedBits operator[](unsigned int idx) const {
        return offset + idx;
    }
    __device__ __forceinline__
    SelectInputIterator operator+(unsigned int idx) {
        return SelectInputIterator(offset + idx);
    }
    unsigned int offset;

};

struct SelectOutput {
    __device__ __forceinline__
    SelectOutput(sq::PackedBits _xBegin, sq::PackedBits &_d_out) : xBegin(_xBegin), d_out(_d_out) { }
    __device__ __forceinline__
    void operator=(sq::PackedBits &v) const {
        d_out = v + xBegin;
    }
    sq::PackedBits xBegin;
    sq::PackedBits &d_out;
};

struct SelectOutputIterator {
    SelectOutputIterator(sq::PackedBits _xBegin, sq::PackedBits *_d_out) : xBegin(_xBegin), d_out(_d_out) { }
    __device__ __forceinline__
    SelectOutput operator[](unsigned int idx) const {
        return SelectOutput(xBegin, d_out[idx]);
    }
    sq::PackedBits xBegin;
    sq::PackedBits *d_out;
};


template<class real> struct SelectOp {
    SelectOp(real _val, const real *_d_vals) : val(_val), d_vals(_d_vals) { }
    __device__ __forceinline__
    bool operator()(const sq::PackedBits &idx) const {
        return val == d_vals[idx];
    }
    real val;
    const real *d_vals;
};

}

namespace std {
template<>
struct iterator_traits<SelectInputIterator> : sqaod_cuda::base_iterator_traits<sq::PackedBits> { };
template<>
struct iterator_traits<SelectOutputIterator> : sqaod_cuda::base_iterator_traits<sq::PackedBits> { };
template<class real>
struct iterator_traits<SelectOp<real> > : sqaod_cuda::base_iterator_traits<real> { };

}


template<class real> void DeviceDenseGraphBatchSearch<real>::
select(sq::PackedBits *d_out, sq::SizeType *d_nOut, sq::PackedBits xBegin, real val, const real *d_vals, sq::SizeType nIn) {
    SelectInputIterator in;
    SelectOutputIterator out(xBegin, d_out);
    SelectOp<real> selectOp(val, d_vals);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          in, out, d_nOut, nIn, selectOp, devStream_->getCudaStream(), CUB_DEBUG);
    // Allocate temporary storage
    d_temp_storage = devStream_->allocate(temp_storage_bytes);
    // Run selection
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          in, out, d_nOut, nIn, selectOp, devStream_->getCudaStream(), CUB_DEBUG);
}



template class sqaod_cuda::DeviceDenseGraphBatchSearch<double>;
template class sqaod_cuda::DeviceDenseGraphBatchSearch<float>;



