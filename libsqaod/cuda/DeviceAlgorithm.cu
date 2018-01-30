#include "DeviceAlgorithm.h"
#include "Device.h"
#include "cudafuncs.h"
#include <cub/cub.cuh>

using namespace sqaod_cuda;
namespace sq = sqaod;

template<class V>
__global__ static
void generateBitsSequenceKernel(V *d_data, int N,
                                sq::SizeType nSeqs, sq::PackedBits xBegin, sq::PackedBits xEnd) {
    sqaod::IdxType seqIdx = blockDim.y * (blockDim.x * blockIdx.x) + blockIdx.y;
    if ((seqIdx < nSeqs) && (threadIdx.x < N)) {
        sq::PackedBits bits = xBegin + seqIdx;
        bool bitSet = (bits >> (N - 1 - threadIdx.x));
        d_data[seqIdx * N + threadIdx.x] = bitSet ? V(1) : V(0);
    }
}


template<class V>
void DeviceAlgorithm::
generateBitsSequence(V *d_data, int N,
                     sqaod::PackedBits xBegin, sqaod::PackedBits xEnd) {
    dim3 blockDim, gridDim;
    blockDim.x = divru(N, 32); /* Packed bits <= 63 bits. */
    blockDim.y = 128 / blockDim.x; /* 2 or 4 */
    sq::PackedBits seqPerBlock = 128 / blockDim.x;
    sq::SizeType nSeqs = xEnd - xBegin;
    gridDim.x = divru(xEnd - xBegin, seqPerBlock);
    generateBitsSequenceKernel<V>
            <<<gridDim, blockDim, 0, stream_>>>(d_data, N, nSeqs, xBegin, xEnd);
    DEBUG_SYNC;
}

namespace {


template<class V>
struct Equal {
    V val_;
    __host__
    Equal(const V val) : val_(val) { }
    __device__ __forceinline__
    bool operator()(const V &val) const {
        return val_ == val;
    }
};

}


template<class real>
void DeviceAlgorithm::
partition_Emin(sqaod::PackedBits *d_bitsListMin, sq::SizeType *d_nMin,
               const real Emin,  const real *d_Ebatch,
               const sqaod::PackedBits *d_bitsList, sqaod::SizeType len) {

    Equal<real> selOp(Emin);
    
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          d_bitsList, d_bitsListMin, d_nMin, len, selOp);
    // Allocate temporary storage
    d_temp_storage = devStream_->allocate(temp_storage_bytes);
    // Run selection
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          d_bitsList, d_bitsListMin, d_nMin, len, selOp);
}

DeviceAlgorithm::DeviceAlgorithm(DeviceStream *devStream) {
    devStream_ = NULL;
    stream_ = NULL;
    if (devStream != NULL)
        assignStream(devStream);
}

void DeviceAlgorithm::assignStream(DeviceStream *devStream) {
    devStream_ = devStream;
    stream_ = devStream->getCudaStream();
}



// Determine temporary device storage requirements

template void DeviceAlgorithm::
generateBitsSequence(double *bitsSequence, int N,
                     sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);
template void DeviceAlgorithm::
generateBitsSequence(float *bitsSequence, int N,
                     sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);
template void DeviceAlgorithm::
generateBitsSequence(char *bitsSequence, int N,
                     sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);


template void DeviceAlgorithm::
partition_Emin(sqaod::PackedBits *bitsListMin, sq::SizeType *d_nMin,
               const double Emin,  const double *d_Ebatch,
               const sqaod::PackedBits *bitsList, sqaod::SizeType len);

template void DeviceAlgorithm::
partition_Emin(sqaod::PackedBits *bitsListMin, sq::SizeType *d_nMin,
               const float Emin,  const float *d_Ebatch,
               const sqaod::PackedBits *bitsList, sqaod::SizeType len);

