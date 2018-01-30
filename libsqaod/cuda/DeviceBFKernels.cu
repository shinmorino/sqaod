#include "DeviceBFKernels.h"
#include "cudafuncs.h"
#include <cub/cub.cuh>
#include <device_launch_parameters.h>

using namespace sqaod_cuda;
namespace sq = sqaod;

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


template<class real> void DeviceBFKernelsType<real>::
generateBitsSequence(real *d_data, int N,
                     sq::PackedBits xBegin, sq::PackedBits xEnd) {
    dim3 blockDim, gridDim;
    blockDim.x = roundUp(N, 32); /* Packed bits <= 63 bits. */
    blockDim.y = 128 / blockDim.x; /* 2 or 4, sequences per block. */
    sq::SizeType nSeqs = sq::SizeType(xEnd - xBegin);
    gridDim.x = (unsigned int)divru((unsigned int)(xEnd - xBegin), blockDim.y);
    generateBitsSequenceKernel
            <<<gridDim, blockDim, 0, stream_>>>(d_data, N, nSeqs, xBegin);
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




template<class real> void DeviceBFKernelsType<real>::
select(sq::PackedBits *d_out, sq::SizeType *d_nOut, sq::PackedBits xBegin, real val, const real *d_vals, sq::SizeType nIn) {
    SelectInputIterator in;
    SelectOutputIterator out(xBegin, d_out);
    SelectOp<real> selectOp(val, d_vals);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          in, out, d_nOut, nIn, selectOp, stream_, CUB_DEBUG);
    // Allocate temporary storage
    d_temp_storage = devStream_->allocate(temp_storage_bytes);
    // Run selection
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          in, out, d_nOut, nIn, selectOp, stream_, CUB_DEBUG);
}


template<class real>
DeviceBFKernelsType<real>::DeviceBFKernelsType(DeviceStream *devStream) {
    devStream_ = NULL;
    stream_ = NULL;
    if (devStream != NULL)
        assignStream(devStream);
}

template<class real>
void DeviceBFKernelsType<real>::assignStream(DeviceStream *devStream) {
    devStream_ = devStream;
    stream_ = devStream->getCudaStream();
}

template class DeviceBFKernelsType<double>;
template class DeviceBFKernelsType<float>;
