#pragma once

#include <common/types.h>
#include <cub/cub.cuh>
#include <cuda/cub_iterator.cuh>
#include <cuda/cudafuncs.h>
#include <cuda/DeviceSegmentedSum.h>
#include <map>

namespace sq = sqaod;

namespace sqaod_cuda {


template<int ITEMS, class V>
__device__ __forceinline__ static
V sumArray(const V *v) {
    /* Max 8 */
    switch (ITEMS) {
    case 1:
        return v[0];
    case 2:
        return v[0] + v[1];
    case 3:
        return v[0] + v[1] + v[2];
    case 4:
        return (v[0] + v[1]) + (v[2] + v[3]);
    case 5:
        return (v[0] + v[1]) + (v[2] + v[3]) + v[4];
    case 6:
        return (v[0] + v[1]) + (v[2] + v[3]) + (v[4] + v[5]);
    case 7:
        return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + v[6]);
    case 8:
        return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
    default:
        break;
    }
    return V();
}

        
enum { WARP_SIZE = 32 };

/* size <= 32 */
template<int BLOCK_DIM, int ITEMS_PER_THREAD, int OUTPUT_PER_SEG, class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_32(InputIterator in, OutputIterator out,
                      OffsetIterator segOffset, sq::SizeType segLen, sq::SizeType nSegments) {

    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;

    int iSubSegment = BLOCK_DIM / WARP_SIZE * blockIdx.x + threadIdx.x / WARP_SIZE;
    int iSegment = iSubSegment / OUTPUT_PER_SEG;
    int iSegIdx = (iSubSegment % OUTPUT_PER_SEG) * WARP_SIZE + threadIdx.x % WARP_SIZE;

    enum { INPUT_STRIDE = WARP_SIZE * OUTPUT_PER_SEG };

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];

        V v[ITEMS_PER_THREAD];
#pragma unroll
        for (int idx = 0; idx < ITEMS_PER_THREAD - 1; ++idx) {
            int inIdx = INPUT_STRIDE * idx + iSegIdx;
            v[idx] = in[segBegin + inIdx];
        }
        int lastSegIdx = iSegIdx + INPUT_STRIDE * (ITEMS_PER_THREAD - 1);
        v[ITEMS_PER_THREAD - 1] = (lastSegIdx < segLen) ? in[segBegin + lastSegIdx] : V();
        sum = sumArray<ITEMS_PER_THREAD>(v);

        typedef cub::WarpReduce<V> WarpReduce;
        __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_DIM / WARP_SIZE];
        sum = WarpReduce(temp_storage[iSubSegment]).Sum(sum);
    }

    if ((threadIdx.x % warpSize) == 0)
        out[iSubSegment] = sum;
}


template<int BLOCK_DIM, int ITEMS_PER_THREAD, int OUTPUT_PER_SEG, class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_64(InputIterator in, OutputIterator out, OffsetIterator segOffset, sq::SizeType segLen,
                      sq::SizeType nSegments) {

    enum {
        N_REDUCE_THREADS = 64,
        WARPS_IN_BLOCK = BLOCK_DIM / WARP_SIZE,
        INPUT_STRIDE = N_REDUCE_THREADS * OUTPUT_PER_SEG,

    };

    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;

    int iSubSegment = (BLOCK_DIM / N_REDUCE_THREADS) * blockIdx.x + (threadIdx.x / N_REDUCE_THREADS);
    int iSegment = iSubSegment / OUTPUT_PER_SEG;
    int warpId = threadIdx.x / WARP_SIZE;
    int iSubSegIdxInBlock = warpId / (BLOCK_DIM / N_REDUCE_THREADS);
    int iSegIdx = (iSubSegment % OUTPUT_PER_SEG) * N_REDUCE_THREADS + threadIdx.x % N_REDUCE_THREADS;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];

        V v[ITEMS_PER_THREAD];
#pragma unroll
        for (int idx = 0; idx < ITEMS_PER_THREAD - 1; ++idx) {
            int inIdx = INPUT_STRIDE * idx + iSegIdx;
            v[idx] = in[segBegin + inIdx];
        }
        int lastSegIdx = iSegIdx + INPUT_STRIDE * (ITEMS_PER_THREAD - 1);
        v[ITEMS_PER_THREAD - 1] = (lastSegIdx < segLen) ? in[segBegin + lastSegIdx] : V();
        sum = sumArray<ITEMS_PER_THREAD>(v);

        typedef cub::WarpReduce<V> WarpReduce;
        __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_IN_BLOCK];
        __shared__ V partialSum[WARPS_IN_BLOCK];
        partialSum[warpId] = WarpReduce(temp_storage[warpId]).Sum(sum);
        V *seqPartialSum = &partialSum[iSubSegIdxInBlock * 2];
        __syncthreads();
        sum = seqPartialSum[0] + seqPartialSum[1];
    }
    if ((threadIdx.x % N_REDUCE_THREADS) == 0)
        out[iSubSegment] = sum;
}


template<int BLOCK_DIM, int ITEMS_PER_THREAD, class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_Block(InputIterator in, OutputIterator out,
                         OffsetIterator segOffset, int segLen, sq::SizeType nSegments) {
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;

    int iSegment = blockIdx.x;
    int iSegIdx = threadIdx.x;

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];

        V v[ITEMS_PER_THREAD];
#pragma unroll
        for (int idx = 0; idx < ITEMS_PER_THREAD - 1; ++idx)
            v[idx] = in[segBegin + iSegIdx + BLOCK_DIM * idx];
        int lastSegIdx = iSegIdx + BLOCK_DIM * (ITEMS_PER_THREAD - 1);
        v[ITEMS_PER_THREAD - 1] = (lastSegIdx < segLen) ? in[segBegin + lastSegIdx] : V();
        sum = sumArray<ITEMS_PER_THREAD>(v);

        typedef cub::BlockReduce<V, BLOCK_DIM> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        sum = BlockReduce(temp_storage).Sum(sum);
    }
    if (threadIdx.x == 0)
        out[iSegment] = sum;
}


template<int BLOCK_DIM, int ITEMS_PER_THREAD, int OUTPUT_PER_SEG, 
         class InputIterator, class OutputIterator, class OffsetIterator>
__global__ static void
segmentedSumKernel_Striped(InputIterator in, OutputIterator out,
                           OffsetIterator segOffset, int segLen, sq::SizeType nSegments) {
    typedef typename std::iterator_traits<OutputIterator>::value_type V;
    typedef typename std::iterator_traits<OffsetIterator>::value_type OffsetT;

    int iSegment = blockIdx.x / OUTPUT_PER_SEG;
    int iSegBlock = blockIdx.x % OUTPUT_PER_SEG;
    int iSegIdx = BLOCK_DIM * iSegBlock + threadIdx.x;

    enum { INPUT_STRIDE = BLOCK_DIM * OUTPUT_PER_SEG };

    V sum = V();
    if (iSegment < nSegments) {
        OffsetT segBegin = segOffset[iSegment];

        V v[ITEMS_PER_THREAD];
#pragma unroll
        for (int idx = 0; idx < ITEMS_PER_THREAD - 1; ++idx) {
            int inIdx = INPUT_STRIDE * idx + iSegIdx;
            v[idx] = in[segBegin + inIdx];
        }
        int lastSegIdx = iSegIdx + INPUT_STRIDE * (ITEMS_PER_THREAD - 1);
        v[ITEMS_PER_THREAD - 1] = (lastSegIdx < segLen) ? in[segBegin + lastSegIdx] : V();
        sum = sumArray<ITEMS_PER_THREAD>(v);

        typedef cub::BlockReduce<V, BLOCK_DIM> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        sum = BlockReduce(temp_storage).Sum(sum);
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}


template<class V, class InputIterator, class OutputIterator, class OffsetIterator>
struct DeviceSegmentedSumTypeImpl : DeviceSegmentedSumType<V> {

    typedef DeviceSegmentedSumTypeImpl<V, InputIterator, OutputIterator, OffsetIterator> SelfType;
    typedef DeviceSegmentedSumType<V> Base;

    using Base::d_tempStoragePreAlloc_;
    using Base::tempStorageSize_;
    using Base::d_tempStorage_;
    using Base::devAlloc_;
    using Base::devStream_;
    using Base::segLen_;
    using Base::nSegments_;

    DeviceSegmentedSumTypeImpl(Device &device, DeviceStream *devStream = NULL) : Base(device, devStream) {
        registerMethods();
        stream_ = devStream_->getCudaStream();
    }

    DeviceSegmentedSumTypeImpl(DeviceStream *devStream) : Base(devStream) {
        registerMethods();
        stream_ = devStream_->getCudaStream();
    }

    typedef void (SelfType::*SumMethod)(InputIterator in, OutputIterator out, OffsetIterator segOffset);

    void operator()(InputIterator in, OutputIterator out, OffsetIterator segOffset) {
        if (d_tempStoragePreAlloc_ != NULL)
            d_tempStorage_ = d_tempStoragePreAlloc_;
        else if (tempStorageSize_ != 0)
            devStream_->allocate(&d_tempStorage_, tempStorageSize_);
        (this->*sumMethod_)(in, out, segOffset);
    }


    template<int ITEMS_PER_THREAD> void
    segmentedSum_32(InputIterator in, OutputIterator out, OffsetIterator segOffset) {

        enum { BLOCK_DIM = 128 };

        dim3 blockDim(BLOCK_DIM);
        dim3 gridDim(divru(nSegments_, 4u));
        segmentedSumKernel_32<BLOCK_DIM, ITEMS_PER_THREAD, 1>
            <<<gridDim, blockDim, 0, stream_>>>(in, out, segOffset, segLen_, nSegments_);
        DEBUG_SYNC;
    }

    template<int ITEMS_PER_THREAD> void
    segmentedSum_64(InputIterator in, OutputIterator out, OffsetIterator segOffset) {
        enum { BLOCK_DIM = 128 };

        dim3 blockDim(BLOCK_DIM);
        dim3 gridDim(divru(nSegments_, 2u));
        segmentedSumKernel_64<BLOCK_DIM, ITEMS_PER_THREAD, 1>
            <<<gridDim, blockDim, 0, stream_>>>(in, out, segOffset, segLen_, nSegments_);
        DEBUG_SYNC;
    }

    template<int BLOCK_DIM, int ITEMS_PER_THREAD> void
    segmentedSum_Block(InputIterator in, OutputIterator out, OffsetIterator segOffset) {
        dim3 blockDim(BLOCK_DIM);
        dim3 gridDim(nSegments_);
        segmentedSumKernel_Block<BLOCK_DIM, ITEMS_PER_THREAD>
            <<<gridDim, blockDim, 0, stream_>>>(in, out, segOffset, segLen_, nSegments_);
        DEBUG_SYNC;
    }

    template<int N_REDUCE_THREADS, int ITEMS_PER_THREAD, int OUTPUT_PER_SEG> void
    segmentedSum_2step(InputIterator in, OutputIterator out, OffsetIterator segOffset) {
        if (N_REDUCE_THREADS == 32) {
            enum { BLOCK_DIM = 128 };
            dim3 blockDim(BLOCK_DIM);
            dim3 gridDim(divru(nSegments_ * OUTPUT_PER_SEG, 4u));
            segmentedSumKernel_32<BLOCK_DIM, ITEMS_PER_THREAD, OUTPUT_PER_SEG>
                <<<gridDim, blockDim, 0, stream_>>>(in, d_tempStorage_, segOffset, segLen_, nSegments_);
            DEBUG_SYNC;
        }
        else if (N_REDUCE_THREADS == 64) {
            enum { BLOCK_DIM = 128 };
            dim3 blockDim(BLOCK_DIM);
            dim3 gridDim(divru(nSegments_ * OUTPUT_PER_SEG, 2u));
            segmentedSumKernel_64<BLOCK_DIM, ITEMS_PER_THREAD, OUTPUT_PER_SEG>
                <<<gridDim, blockDim, 0, stream_>>>(in, d_tempStorage_, segOffset, segLen_, nSegments_);
            DEBUG_SYNC;
        }
        else {
            enum { BLOCK_DIM = N_REDUCE_THREADS };
            dim3 blockDim(BLOCK_DIM);
            dim3 gridDim(nSegments_ * OUTPUT_PER_SEG);
            segmentedSumKernel_Striped<BLOCK_DIM, ITEMS_PER_THREAD, OUTPUT_PER_SEG>
                <<<gridDim, blockDim, 0, stream_>>>(in, d_tempStorage_, segOffset, segLen_, nSegments_);
            DEBUG_SYNC;
        }

        enum { BLOCK_DIM_32 = 128 };
        dim3 blockDim(BLOCK_DIM_32);
        dim3 gridDim(divru(nSegments_, sq::SizeType(BLOCK_DIM_32 / WARP_SIZE)));
        segmentedSumKernel_32<BLOCK_DIM_32, 1, 1>
            <<<gridDim, blockDim, 0, stream_>>>(d_tempStorage_, out, Linear(OUTPUT_PER_SEG, 0), OUTPUT_PER_SEG, nSegments_);
        DEBUG_SYNC;
    }

    void reg(int base, int nItems, SumMethod method) {
        methodMap_[base * nItems] = method;
    }

    void registerMethods() {
        reg(32, 1, &SelfType::segmentedSum_32<1>);
        reg(32, 2, &SelfType::segmentedSum_32<2>);
        reg(32, 3, &SelfType::segmentedSum_32<3>);
        reg(32, 4, &SelfType::segmentedSum_32<4>);
        reg(32, 5, &SelfType::segmentedSum_32<5>);
        reg(32, 6, &SelfType::segmentedSum_32<6>);
        reg(32, 7, &SelfType::segmentedSum_32<7>);
        reg(32, 8, &SelfType::segmentedSum_32<8>);

        //reg(64, 1, &SelfType::segmentedSum_64<1>);
        //reg(64, 2, &SelfType::segmentedSum_64<2>);
        //reg(64, 3, &SelfType::segmentedSum_64<3>);
        //reg(64, 4, &SelfType::segmentedSum_64<4>);
        reg(64, 5, &SelfType::segmentedSum_64<5>);
        reg(64, 6, &SelfType::segmentedSum_64<6>);
        reg(64, 7, &SelfType::segmentedSum_64<7>);
        reg(64, 8, &SelfType::segmentedSum_64<8>);

        //reg(128, 1, &SelfType::segmentedSum_Block<128, 1>);
        //reg(128, 2, &SelfType::segmentedSum_Block<128, 2>);
        //reg(128, 3, &SelfType::segmentedSum_Block<128, 3>);
        //reg(128, 4, &SelfType::segmentedSum_Block<128, 4>);
        reg(128, 5, &SelfType::segmentedSum_Block<128, 5>);
        reg(128, 6, &SelfType::segmentedSum_Block<128, 6>);
        reg(128, 7, &SelfType::segmentedSum_Block<128, 7>);
        reg(128, 8, &SelfType::segmentedSum_Block<128, 8>);

        //reg(256, 1, &SelfType::segmentedSum_Block<256, 1>);
        //reg(256, 2, &SelfType::segmentedSum_Block<256, 2>);
        //reg(256, 3, &SelfType::segmentedSum_Block<256, 3>);
        //reg(256, 4, &SelfType::segmentedSum_Block<256, 4>);
        reg(256, 5, &SelfType::segmentedSum_Block<256, 5>);
        reg(256, 6, &SelfType::segmentedSum_Block<256, 6>);
        reg(256, 7, &SelfType::segmentedSum_Block<256, 7>);
        reg(256, 8, &SelfType::segmentedSum_Block<256, 8>);

        //reg(512, 1, &SelfType::segmentedSum_Block<512, 1>);
        //reg(512, 2, &SelfType::segmentedSum_Block<512, 2>);
        //reg(512, 3, &SelfType::segmentedSum_Block<512, 3>);
        //reg(512, 4, &SelfType::segmentedSum_Block<512, 4>);
        reg(512, 5, &SelfType::segmentedSum_Block<512, 5>);
        reg(512, 6, &SelfType::segmentedSum_Block<512, 6>);
        reg(512, 7, &SelfType::segmentedSum_Block<512, 7>);
        reg(512, 8, &SelfType::segmentedSum_Block<512, 8>);

        reg(1024, 5, &SelfType::segmentedSum_2step<32, 5, 32>);
        reg(1024, 6, &SelfType::segmentedSum_2step<32, 6, 32>);
        reg(1024, 7, &SelfType::segmentedSum_2step<32, 7, 32>);
        reg(1024, 8, &SelfType::segmentedSum_2step<32, 8, 32>);

        reg(2048, 5, &SelfType::segmentedSum_2step<64, 5, 32>);
        reg(2048, 6, &SelfType::segmentedSum_2step<64, 6, 32>);
        reg(2048, 7, &SelfType::segmentedSum_2step<64, 7, 32>);
        reg(2048, 8, &SelfType::segmentedSum_2step<64, 8, 32>);

#if 0
        reg(4096, 5, &SelfType::segmentedSum_2step<128, 5, 32>);
        reg(4096, 6, &SelfType::segmentedSum_2step<128, 6, 32>);
        reg(4096, 7, &SelfType::segmentedSum_2step<128, 7, 32>);
        reg(4096, 8, &SelfType::segmentedSum_2step<128, 8, 32>);

        reg(8192, 5, &SelfType::segmentedSum_2step<256, 5, 32>);
        reg(8192, 6, &SelfType::segmentedSum_2step<256, 6, 32>);
        reg(8192, 7, &SelfType::segmentedSum_2step<256, 7, 32>);
        reg(8192, 8, &SelfType::segmentedSum_2step<256, 8, 32>);
#endif    
    }

    virtual void chooseKernel() {
        typename MethodMap::iterator it = methodMap_.lower_bound(segLen_);
        throwErrorIf(it == methodMap_.end(), "Segment length (%d) not supported.", segLen_);
        sumMethod_ = it->second;
    }
    typedef std::map<sq::SizeType, SumMethod> MethodMap;
    MethodMap methodMap_;
    SumMethod sumMethod_;
    cudaStream_t stream_;
};

}
