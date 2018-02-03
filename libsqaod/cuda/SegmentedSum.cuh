#pragma once

#include <cub/cub.cuh>

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
segmentedSum_128(InputIterator in,
                 OffsetIterator segOffset, sq::SizeType segLen,
                 sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<InputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;
    typedef cub::BlockReduce<V, 128> BlockReduce;

    int iSegment = blockIdx.x;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];
        OffsetT inPos = segBegin + threadIdx.x;
        V v = threadIdx.x < segLen ? in[inPos] : V();
        sum = BlockReduce(temp_storage).Sum(v);
    }
    return sum;
}


/* 64 < size */
template<class InputIterator, class OffsetIterator>
__device__ __forceinline__ static
typename std::iterator_traits<InputIterator>::value_type
segmentedSum_128Loop(InputIterator in,
                     OffsetIterator segOffset, sq::SizeType segLen,
                     sq::SizeType nBlocksPerSeg, sq::SizeType nSegments) {
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
        for (int idx = BLOCK_DIM * iBlock + threadIdx.x; idx < segLen;
             idx += nBlocksPerSeg * BLOCK_DIM)
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
    int iSeg = (BLOCK_DIM / 64) * blockIdx.x + (threadIdx.x / 64);
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_64(in, segOffset, segLen, nSegments);
    if ((threadIdx.x % 64) == 0)
        out[iSeg] = sum;
}

template<class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_128(InputIterator in, OutputIterator out,
                       OffsetIterator segOffset, sq::SizeType segLen,
                       sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_128(in, segOffset, segLen, nSegments);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

template<class InputIterator, class OutputIterator,
         class OffsetBeginIterator, class OffsetEndIterator>
__global__ static void
segmentedSumKernel_128Loop(InputIterator in, OutputIterator out,
                           OffsetBeginIterator offsetBegin, OffsetEndIterator offsetEnd,
                           sq::SizeType nBlocksPerSeg,
                           sq::SizeType nSegments) {
    enum { BLOCK_DIM = 128 };
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    V sum = segmentedSum_128Loop(in, offsetBegin, offsetEnd, nBlocksPerSeg, nSegments);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}


template<class InputIterator, class OutputIterator, class OffsetIterator>
void segmentedSum(void *temp_storage, sq::SizeType *temp_storage_bytes,
                  InputIterator in, OutputIterator out,
                  OffsetIterator segOffset, int segLen,
                  sq::SizeType nSegments, int devNo, cudaStream_t stream) {
    *temp_storage_bytes = 0;
    if (segLen <= WARP_SIZE) {
        dim3 blockDim(128);
        dim3 gridDim(divru(nSegments, 4u));
        segmentedSumKernel_32<<<gridDim, blockDim, 0, stream>>>
                (in, out, segOffset, segLen, nSegments);
        DEBUG_SYNC;
    }
    else if (segLen <= WARP_SIZE * 2) {
        dim3 blockDim(128);
        dim3 gridDim(divru(nSegments, 4u));
        segmentedSumKernel_64<<<gridDim, blockDim, 0, stream>>>
                (in, out, segOffset, segLen, nSegments);
        DEBUG_SYNC;
    }
    else if (segLen <= WARP_SIZE * 4) {
        dim3 blockDim(128);
        dim3 gridDim(nSegments);
        segmentedSumKernel_128<<<gridDim, blockDim, 0, stream>>>
                (in, out, segOffset, segLen, nSegments);
        DEBUG_SYNC;
    }
    else {
        /* General case */
        int nThreadsToFillDevice = getNumThreadsToFillDevice(devNo);
        int nThreads = divru(segLen, 128) * nSegments;
        int nLoops = divru(nThreads, nThreadsToFillDevice);
        if (2 <= nLoops) {
            dim3 blockDim(128);
            dim3 gridDim(nSegments);
            segmentedSumKernel_128Loop<<<gridDim, blockDim, 0, stream>>>
                    (in, out, segOffset, segLen, nSegments, 1);
            DEBUG_SYNC;
        }
        else {
            int nBlocksPerSeg = divru(nThreadsToFillDevice, nThreads);
            dim3 blockDim(128);
            dim3 gridDim(nSegments * nBlocksPerSeg);
            typedef typename std::iterator_traits<OutputIterator>::value_type V;
            if (temp_storage == NULL) {
                *temp_storage_bytes = sizeof(V) * nBlocksPerSeg * nSegments;
                return;
            }
            segmentedSumKernel_128Loop<<<gridDim, blockDim, 0, stream>>>
                    (in, (V*)temp_storage, segOffset, segLen, nSegments,
                     nBlocksPerSeg);
            DEBUG_SYNC;
            segmentedSum(NULL, NULL, (V*)temp_storage, out,
                         Linear(nBlocksPerSeg, 0), nBlocksPerSeg, nSegments, devNo, stream);
        }
    }
}

}
