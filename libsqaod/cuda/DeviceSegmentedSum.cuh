#pragma once

#include <cub/cub.cuh>
#include <cuda/cub_iterator.cuh>
#include <cuda/cudafuncs.h>

namespace sqaod_cuda {

enum { WARP_SIZE = 32 };

/* size <= 32 */
template<class InputIterator, class OffsetIterator>
__device__ __forceinline__ static
typename std::iterator_traits<InputIterator>::value_type
segmentedSum_32(InputIterator in,
                OffsetIterator segOffset, sq::SizeType segLen,
                sq::SizeType nSegments) {
    typedef typename std::iterator_traits<InputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;
    typedef cub::WarpReduce<V> WarpReduce;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int iSegment = gid / warpSize;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];
        int laneId = gid % warpSize;
        OffsetT inPos = segBegin + laneId;
        V v = (laneId < segLen) ? in[inPos] : V();
        __shared__ typename WarpReduce::TempStorage temp_storage;
        sum = WarpReduce(temp_storage).Sum(v);
    }
    return sum;
}

/* 32 < size <= 64 */
template<class InputIterator, class OffsetIterator>
__device__ __forceinline__ static
typename std::iterator_traits<InputIterator>::value_type
segmentedSum_64(InputIterator in,
                OffsetIterator segOffset, sq::SizeType segLen,
                sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<InputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;
    typedef cub::WarpReduce<V> WarpReduce;
    int iSegment = (BLOCK_DIM / WARP_SIZE) * blockIdx.x + threadIdx.x / WARP_SIZE;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];
        int laneId = threadIdx.x % WARP_SIZE;
        OffsetT inPos = segBegin + laneId;
        V v = in[inPos];
        v += ((laneId + warpSize) < segLen) ? in[inPos + warpSize] : V();
        __shared__ typename WarpReduce::TempStorage temp_storage;
        sum = WarpReduce(temp_storage).Sum(v);
    }
    return sum;
}


/* 64 < size */
template<class InputIterator, class OffsetIterator>
__device__ __forceinline__ static
typename std::iterator_traits<InputIterator>::value_type
segmentedSum_128Striped(InputIterator in,
                        OffsetIterator segOffset, sq::SizeType segLen,
                        sq::SizeType nSegments, sq::SizeType nBlocksPerSeg) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<InputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;
    typedef cub::BlockReduce<V, BLOCK_DIM> BlockReduce;

    int iSegment = blockIdx.x / nBlocksPerSeg;
    int iBlock = blockIdx.x % nBlocksPerSeg;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];
        V v = V();
        for (int idx = BLOCK_DIM * iBlock + threadIdx.x; idx < segLen; idx += nBlocksPerSeg * BLOCK_DIM)
            v += in[segBegin + idx];
        sum = BlockReduce(temp_storage).Sum(v);
    }
    return sum;
}

/* 64 < size */
template<class InputIterator, class OffsetIterator>
__device__ __forceinline__ static
typename std::iterator_traits<InputIterator>::value_type
segmentedSum_128x8(InputIterator in,
                   OffsetIterator segOffset, sq::SizeType segLen,
                   sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<InputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;
    typedef cub::BlockReduce<V, BLOCK_DIM> BlockReduce;

    int iSegment = blockIdx.x;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment] + threadIdx.x;
        V v = V();
#pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            if (threadIdx.x + BLOCK_DIM * idx < segLen)
                v += in[segBegin + BLOCK_DIM * idx];
        }
        sum = BlockReduce(temp_storage).Sum(v);
    }
    return sum;
}

template<class InputIterator, class OffsetIterator>
__device__ __forceinline__ static
typename std::iterator_traits<InputIterator>::value_type
segmentedSum_64x16(InputIterator in,
                   OffsetIterator segOffset, sq::SizeType segLen,
                   sq::SizeType nSegments) {
    enum { BLOCK_DIM = 64 };
    typedef typename std::iterator_traits<InputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;
    typedef cub::BlockReduce<V, BLOCK_DIM> BlockReduce;

    int iSegment = blockIdx.x;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment] + threadIdx.x;
        V v = V();
#pragma unroll
        for (int idx = 0; idx < 15; ++idx) {
            v += in[segBegin + BLOCK_DIM * idx];
        }
        if (15 * BLOCK_DIM + threadIdx.x < segLen)
            v += in[segBegin + 15 * BLOCK_DIM];
        sum = BlockReduce(temp_storage).Sum(v);
    }
    return sum;
}


template<class InputIterator, class OffsetIterator>
__device__ __forceinline__ static
typename std::iterator_traits<InputIterator>::value_type
segmentedSum_512(InputIterator in,
                  OffsetIterator segOffset, sq::SizeType segLen, sq::SizeType nSegments) {
    enum { BLOCK_DIM = 512 };
    typedef typename std::iterator_traits<InputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;
    typedef cub::BlockReduce<V, BLOCK_DIM> BlockReduce;

    int iSegment = blockIdx.x;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];
        V v = V();
        for (int idx = threadIdx.x; idx < segLen; idx += BLOCK_DIM)
            v += in[segBegin + idx];
        sum = BlockReduce(temp_storage).Sum(v);
    }
    return sum;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_32(InputIterator in, OutputIterator out,
                      OffsetIterator segOffset, sq::SizeType segLen,
                      sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    int iSeg = BLOCK_DIM / 32 * blockIdx.x + threadIdx.x / 32;
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_32(in, segOffset, segLen, nSegments);
    if ((threadIdx.x % warpSize) == 0)
        out[iSeg] = sum;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_64(InputIterator in, OutputIterator out,
                      OffsetIterator segOffset, sq::SizeType segLen,
                      sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_64(in, segOffset, segLen, nSegments);
    if ((threadIdx.x % WARP_SIZE) == 0) {
        int iSeg = (BLOCK_DIM / WARP_SIZE) * blockIdx.x + (threadIdx.x / WARP_SIZE);
        out[iSeg] = sum;
    }
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_128(InputIterator in, OutputIterator out,
                       OffsetIterator segOffset, sq::SizeType segLen,
                       sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_128Striped(in, segOffset, segLen, nSegments, 1);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

template<class InputIterator, class OutputIterator,
         class OffsetBeginIterator, class OffsetEndIterator>
__global__ static void
segmentedSumKernel_128Striped(InputIterator in, OutputIterator out,
                              OffsetBeginIterator offsetBegin, OffsetEndIterator offsetEnd,
                              sq::SizeType nSegments, sq::SizeType nBlocksPerSeg) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_128Striped(in, offsetBegin, offsetEnd, nSegments, nBlocksPerSeg);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}


template<class InputIterator, class OutputIterator,
         class OffsetBeginIterator, class OffsetEndIterator>
__global__ static void
segmentedSumKernel_64x16(InputIterator in, OutputIterator out,
                         OffsetBeginIterator offsetBegin, OffsetEndIterator offsetEnd,
                         sq::SizeType nSegments) {
    enum { BLOCK_DIM = 64 };
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_64x16(in, offsetBegin, offsetEnd, nSegments);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

template<class InputIterator, class OutputIterator,
         class OffsetBeginIterator, class OffsetEndIterator>
__global__ static void
segmentedSumKernel_128x8(InputIterator in, OutputIterator out,
                         OffsetBeginIterator offsetBegin, OffsetEndIterator offsetEnd,
                         sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_128x8(in, offsetBegin, offsetEnd, nSegments);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

template<class InputIterator, class OutputIterator,
         class OffsetBeginIterator, class OffsetEndIterator>
__global__ static void
segmentedSumKernel_512Loop(InputIterator in, OutputIterator out,
                            OffsetBeginIterator offsetBegin, OffsetEndIterator offsetEnd,
                            sq::SizeType nSegments) {
    enum { BLOCK_DIM = 512};
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_512(in, offsetBegin, offsetEnd, nSegments);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}


template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum_32(InputIterator in, OutputIterator out,
                     OffsetIterator segOffset, int segLen, sq::SizeType nSegments, cudaStream_t stream) {
    assert(segLen <= 32);
    dim3 blockDim(128);
    dim3 gridDim(divru(nSegments, 4u));
    segmentedSumKernel_32<<<gridDim, blockDim, 0, stream>>>
            (in, out, segOffset, segLen, nSegments);
    DEBUG_SYNC;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum_64(InputIterator in, OutputIterator out,
                     OffsetIterator segOffset, int segLen, sq::SizeType nSegments, cudaStream_t stream) {
    assert((32 < segLen) && (segLen <= 64));
    dim3 blockDim(128);
    dim3 gridDim(divru(nSegments, 4u));
    segmentedSumKernel_64<<<gridDim, blockDim, 0, stream>>>
            (in, out, segOffset, segLen, nSegments);
    DEBUG_SYNC;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum_128(InputIterator in, OutputIterator out,
                      OffsetIterator segOffset, int segLen, sq::SizeType nSegments, cudaStream_t stream) {
    assert((64 < segLen) && (segLen <= 128));
    dim3 blockDim(128);
    dim3 gridDim(nSegments);
    segmentedSumKernel_128<<<gridDim, blockDim, 0, stream>>>(in, out, segOffset, segLen, nSegments);
    DEBUG_SYNC;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum_64x16(InputIterator in, OutputIterator out,
                        OffsetIterator segOffset, int segLen, sq::SizeType nSegments, cudaStream_t stream) {
    dim3 blockDim(64);
    dim3 gridDim(nSegments);
    segmentedSumKernel_64x16<<<gridDim, blockDim, 0, stream>>>(in, out, segOffset, segLen, nSegments);
    DEBUG_SYNC;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum_128x8(InputIterator in, OutputIterator out,
                        OffsetIterator segOffset, int segLen, sq::SizeType nSegments, cudaStream_t stream) {
    dim3 blockDim(128);
    dim3 gridDim(nSegments);
    segmentedSumKernel_128x8<<<gridDim, blockDim, 0, stream>>>(in, out, segOffset, segLen, nSegments);
    DEBUG_SYNC;
}


template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum_128Loop(InputIterator in, OutputIterator out,
                          OffsetIterator segOffset, int segLen, sq::SizeType nSegments, sq::SizeType nBlocksPerSeq, cudaStream_t stream) {
    assert(128 < segLen);
    dim3 blockDim(128);
    dim3 gridDim(nSegments * nBlocksPerSeq);
    segmentedSumKernel_128Striped<<<gridDim, blockDim, 0, stream>>>(in, out, segOffset, segLen, nSegments, nBlocksPerSeq);
    DEBUG_SYNC;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum_512Loop(InputIterator in, OutputIterator out,
                           OffsetIterator segOffset, int segLen, sq::SizeType nSegments, cudaStream_t stream) {
    dim3 blockDim(512);
    dim3 gridDim(nSegments);
    segmentedSumKernel_512Loop<<<gridDim, blockDim, 0, stream>>>(in, out, segOffset, segLen, nSegments);
    DEBUG_SYNC;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum(InputIterator in, OutputIterator out,
                  OffsetIterator segOffset, int segLen,
                  sq::SizeType nSegments, cudaStream_t stream) {
    if (segLen <= WARP_SIZE) {
        segmentedSum_32(in, out, segOffset, segLen, nSegments, stream);
    }
    else if (segLen <= WARP_SIZE * 2) {
        segmentedSum_64(in, out, segOffset, segLen, nSegments, stream);
    }
    else if (segLen <= WARP_SIZE * 4) {
        segmentedSum_128(in, out, segOffset, segLen, nSegments, stream);
    }
    else if (segLen <= WARP_SIZE * 8) {
        segmentedSum_128Loop(in, out, segOffset, segLen, nSegments, 1, stream);
    }
    else {
        segmentedSum_512Loop(in, out, segOffset, segLen, nSegments, stream);
    }
}



template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum(void *temp_storage, sq::SizeType *temp_storage_bytes,
                  InputIterator in, OutputIterator out,
                  OffsetIterator segOffset, int segLen,
                  sq::SizeType nSegments, int nThreadsToFillDevice, cudaStream_t stream) {
    if (temp_storage_bytes != NULL)
        *temp_storage_bytes = 0;
    if (segLen <= WARP_SIZE) {
        segmentedSum_32(in, out, segOffset, segLen, nSegments, stream);
    }
    else if (segLen <= WARP_SIZE * 2) {
        segmentedSum_64(in, out, segOffset, segLen, nSegments, stream);
    }
    else if (segLen <= WARP_SIZE * 4) {
        segmentedSum_128(in, out, segOffset, segLen, nSegments, stream);
    }
    else if (segLen <= WARP_SIZE * 64) {
        // segmentedSum_512Loop(in, out, segOffset, segLen, nSegments, stream);
        segmentedSum_128x8(in, out, segOffset, segLen, nSegments, stream);
        // segmentedSum_64x16(in, out, segOffset, segLen, nSegments, stream);
    }
    else {
        /* General case */
        int nThreads = roundUp(segLen, 128) * nSegments;
        int nLoops = divru(nThreads, nThreadsToFillDevice);
        if (2 <= nLoops) {
            segmentedSum_128(in, out, segOffset, segLen, nSegments, stream);
        }
        else {
            assert(temp_storage_bytes != NULL);
            int nBlocksPerSeg = divru(nThreadsToFillDevice, nThreads);
            typedef typename std::iterator_traits<OutputIterator>::value_type V;
            if (temp_storage == NULL) {
                *temp_storage_bytes = sizeof(V) * nBlocksPerSeg * nSegments;
                return;
            }
            dim3 blockDim(128);
            dim3 gridDim(nSegments * nBlocksPerSeg);
            segmentedSum_128Loop(in, (V*)temp_storage, segOffset, segLen, nSegments, nBlocksPerSeg, stream);
            /* Expected max 4096 output from the first stage. */
            assert(nBlocksPerSeg <= 4096);
            segmentedSum((V*)temp_storage, out, Linear(nBlocksPerSeg, 0), nBlocksPerSeg, nSegments, stream);
        }
    }
}

}
