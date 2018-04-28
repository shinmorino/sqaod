#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include "DeviceBipartiteGraphBatchSearch.h"
#include "cub_iterator.cuh"

using namespace sqaod_cuda;

template<class real>
DeviceBipartiteGraphBatchSearch<real>::DeviceBipartiteGraphBatchSearch() {
    N0_ = N1_ = -1;
}


template<class real>
void DeviceBipartiteGraphBatchSearch<real>::assignDevice(Device &device, DeviceStream *devStream) {
    devStream_ = devStream;
    devFormulas_.assignDevice(device, devStream_);
    devCopy_.assignDevice(device, devStream_);
    devAlloc_ = device.objectAllocator();
}

template<class real>
void DeviceBipartiteGraphBatchSearch<real>::deallocate() {
    devAlloc_->deallocate(d_b0_);
    devAlloc_->deallocate(d_b1_);
    devAlloc_->deallocate(d_W_);
    devAlloc_->deallocate(d_bitsMat0_);
    devAlloc_->deallocate(d_bitsMat1_);
    devAlloc_->deallocate(d_Ebatch_);
    devAlloc_->deallocate(d_minXPairs_);

    HostObjectAllocator halloc;
    halloc.deallocate(h_nMinXPairs_);
    halloc.deallocate(h_Emin_);
}


template<class real>
void DeviceBipartiteGraphBatchSearch<real>::
setQUBO(const HostVector &b0, const HostVector &b1,
        const HostMatrix &W,
        sq::SizeType tileSize0, sq::SizeType tileSize1) {
    if (N0_ != -1)
        deallocate();

    N0_ = b0.size;
    N1_ = b1.size;
    devCopy_(&d_b0_, b0);
    devCopy_(&d_b1_, b1);
    devCopy_(&d_W_, W);
    tileSize0_ = tileSize0;
    tileSize1_ = tileSize1;
    minXPairsSize_ = tileSize0 * tileSize1;
    devAlloc_->allocate(&d_bitsMat0_, tileSize0, W.cols);
    devAlloc_->allocate(&d_bitsMat1_, tileSize1, W.rows);
    devAlloc_->allocate(&d_Ebatch_, tileSize1, tileSize0);
    devAlloc_->allocate(&d_minXPairs_, tileSize1 * tileSize0);

    HostObjectAllocator halloc;
    halloc.allocate(&h_nMinXPairs_);
    halloc.allocate(&h_Emin_);
}


template<class real>
void DeviceBipartiteGraphBatchSearch<real>::
calculate_E(sq::PackedBitSet xBegin0, sq::PackedBitSet xEnd0,
            sq::PackedBitSet xBegin1, sq::PackedBitSet xEnd1) {
    xBegin0_ = xBegin0;
    xBegin1_ = xBegin1;
    sq::SizeType nBatch0 = sq::SizeType(xEnd0 - xBegin0);
    sq::SizeType nBatch1 = sq::SizeType(xEnd1 - xBegin1);
    abortIf(tileSize0_ < nBatch0,
            "nBatch0 is too large, tileSize0=%d, nBatch0=%d", int(tileSize0_), int(nBatch0));
    abortIf(tileSize0_ < nBatch0,
            "nBatch1 is too large, tileSize1=%d, nBatch1=%d", int(tileSize1_), int(nBatch1));
    /* FIXME: use stream if effective */
    generateBitsSequence(&d_bitsMat0_, xBegin0, xEnd0);
    generateBitsSequence(&d_bitsMat1_, xBegin1, xEnd1);
    devFormulas_.calculate_E_2d(&d_Ebatch_, d_b0_, d_b1_, d_W_, d_bitsMat0_, d_bitsMat1_);
    devFormulas_.devMath.min(&h_Emin_, d_Ebatch_);
}


template<class real>
void DeviceBipartiteGraphBatchSearch<real>::partition_minXPairs(bool append) {
    assert(d_Ebatch_.dim() == sq::Dim(tileSize1_, tileSize0_));
    if (!append) {
        d_minXPairs_.size = 0;
        select(d_minXPairs_.d_data, h_nMinXPairs_.d_data,
               xBegin0_, xBegin1_, *h_Emin_.d_data,
               d_Ebatch_.d_data, d_Ebatch_.stride, tileSize0_, tileSize1_);
        synchronize();
        d_minXPairs_.size = *h_nMinXPairs_.d_data; /* sync field */
    }
    else if (d_minXPairs_.size < minXPairsSize_) {
        /* append */
        select(&d_minXPairs_.d_data[d_minXPairs_.size], h_nMinXPairs_.d_data,
               xBegin0_, xBegin1_, *h_Emin_.d_data,
               d_Ebatch_.d_data, d_Ebatch_.stride, tileSize0_, tileSize1_);
        synchronize();
        d_minXPairs_.size += *h_nMinXPairs_.d_data; /* sync field */
    }
}

template<class real>
void DeviceBipartiteGraphBatchSearch<real>::synchronize() {
    devStream_->synchronize();
}


template<class real>
__global__ static
void generateBitsSequenceKernel(real *d_data, int stride, int N,
                                sq::SizeType nSeqs, sq::PackedBitSet xBegin) {
    sq::IdxType seqIdx = blockDim.y * blockIdx.x + threadIdx.y;
    if ((seqIdx < nSeqs) && (threadIdx.x < N)) {
        sq::PackedBitSet bits = xBegin + seqIdx;
        bool bitSet = bits & (1ull << (N - 1 - threadIdx.x));
        d_data[seqIdx * stride + threadIdx.x] = bitSet ? real(1) : real(0);
    }
}


template<class real> void DeviceBipartiteGraphBatchSearch<real>::
generateBitsSequence(DeviceMatrix *bitsSequences,
                     sq::PackedBitSet xBegin, sq::PackedBitSet xEnd) {
    sq::SizeType N = bitsSequences->cols;
    sq::SizeType stride = bitsSequences->stride;
    dim3 blockDim, gridDim;
    blockDim.x = roundUp(N, 32); /* Packed bits <= 63 bits. */
    blockDim.y = 128 / blockDim.x; /* 2 or 4, sequences per block. */
    sq::SizeType nSeqs = sq::SizeType(xEnd - xBegin);
    gridDim.x = divru((unsigned int)(xEnd - xBegin), blockDim.y);
    generateBitsSequenceKernel<<<gridDim, blockDim, 0, devStream_->getCudaStream()>>>
            (bitsSequences->d_data, stride, N, nSeqs, xBegin);
    DEBUG_SYNC;
}


namespace {

struct SelectInputIterator {
    __device__ __forceinline__
    SelectInputIterator(sq::PackedBitSetPair _xPairOffset, int _tileSize0)
            : xPairOffset(_xPairOffset), tileSize0(_tileSize0) { }
    __host__
    SelectInputIterator(int _tileSize0) { 
        xPairOffset.bits0 = 0;
        xPairOffset.bits1 = 0;
        tileSize0 = _tileSize0;
    }

    __device__ __forceinline__
    sq::PackedBitSetPair operator[](int idx) const {
        sq::PackedBitSetPair pair;
        pair.bits0 = xPairOffset.bits0 + (idx % tileSize0);
        pair.bits1 = xPairOffset.bits1 + (idx / tileSize0);
        return pair;
    }
    __device__ __forceinline__
    SelectInputIterator operator+(int idx) {
        sq::PackedBitSetPair pair;
        pair.bits0 = xPairOffset.bits0 + (idx % tileSize0);
        pair.bits1 = xPairOffset.bits1 + (idx / tileSize0);
        return SelectInputIterator(pair, tileSize0);
    }
    sq::PackedBitSetPair xPairOffset;
    int tileSize0;
};


struct SelectOutput {
    __device__ __forceinline__
    SelectOutput(sq::PackedBitSet _xBegin0, sq::PackedBitSet _xBegin1, sq::PackedBitSetPair &_d_out)
            : xBegin0(_xBegin0), xBegin1(_xBegin1), d_out(_d_out) { }
    __device__ __forceinline__
    void operator=(sq::PackedBitSetPair &v) const {
        sq::PackedBitSetPair pair;
        pair.bits0 = v.bits0 + xBegin0;
        pair.bits1 = v.bits1 + xBegin1;
        d_out = pair;
    }
    sq::PackedBitSet xBegin0;
    sq::PackedBitSet xBegin1;
    sq::PackedBitSetPair &d_out;
};

struct SelectOutputIterator {
    SelectOutputIterator(sq::PackedBitSet _xBegin0, sq::PackedBitSet _xBegin1,
                         sq::PackedBitSetPair *_d_out) : xBegin0(_xBegin0), xBegin1(_xBegin1),
                                                         d_out(_d_out) { }
    __device__ __forceinline__
    SelectOutput operator[](unsigned int idx) const {
        return SelectOutput(xBegin0, xBegin1, d_out[idx]);
    }
    sq::PackedBitSet xBegin0;
    sq::PackedBitSet xBegin1;
    sq::PackedBitSetPair *d_out;
};


template<class real> struct SelectOp {
    SelectOp(real _val, const real *_d_vals, int _stride)
            : val(_val), d_vals(_d_vals), stride(_stride) { }
    __device__ __forceinline__
    bool operator()(const sq::PackedBitSetPair &idx) const {
        return val == d_vals[idx.bits1 * stride + idx.bits0];
    }
    real val;
    const real *d_vals;
    int stride;
};

}

namespace std {
template<>
struct iterator_traits<SelectInputIterator> : sqaod_cuda::base_iterator_traits<sq::PackedBitSetPair> { };
template<>
struct iterator_traits<SelectOutputIterator> : sqaod_cuda::base_iterator_traits<sq::PackedBitSetPair> { };
template<class real>
struct iterator_traits<SelectOp<real> > : sqaod_cuda::base_iterator_traits<real> { };

}


template<class real> void DeviceBipartiteGraphBatchSearch<real>::
select(sq::PackedBitSetPair *d_out, sq::SizeType *d_nOut,
       sq::PackedBitSet xBegin0, sq::PackedBitSet xBegin1, 
       real val, const real *d_vals, sq::SizeType valsStride, sq::SizeType nIn0, sq::SizeType nIn1) {
    SelectInputIterator in(tileSize0_);

    SelectOutputIterator out(xBegin0, xBegin1, d_out);
    SelectOp<real> selectOp(val, d_vals, valsStride);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    sq::SizeType nIn = nIn0 * nIn1;
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          in, out, d_nOut, nIn, selectOp, devStream_->getCudaStream(), CUB_DEBUG);
    // Allocate temporary storage
    d_temp_storage = devStream_->allocate(temp_storage_bytes);
    // Run selection
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
                          in, out, d_nOut, nIn, selectOp, devStream_->getCudaStream(), CUB_DEBUG);
}



template class sqaod_cuda::DeviceBipartiteGraphBatchSearch<double>;
template class sqaod_cuda::DeviceBipartiteGraphBatchSearch<float>;




// template<class real>
// void BGFuncs<real>::batchSearch(real *E, PackedBitSetPairArray *xPairs,
//                                 const EigenDeviceMatrix &b0, const EigenDeviceMatrix &b1, const EigenDeviceMatrix &W,
//                                 PackedBitSet xBegin0, PackedBitSet xEnd0,
//                                 PackedBitSet xBegin1, PackedBitSet xEnd1) {
//     int nBatch0 = int(xEnd0 - xBegin0);
//     int nBatch1 = int(xEnd1 - xBegin1);

//     real Emin = *E;
//     int N0 = W.cols();
//     int N1 = W.rows();
//     EigenDeviceMatrix eBitsSeq0(nBatch0, N0);
//     EigenDeviceMatrix eBitsSeq1(nBatch1, N1);

//     createBitsSequence(eBitsSeq0.data(), N0, xBegin0, xEnd0);
//     createBitsSequence(eBitsSeq1.data(), N1, xBegin1, xEnd1);
    
//     EigenDeviceMatrix eEBatch = eBitsSeq1 * (W * eBitsSeq0.transpose());
//     eEBatch.rowwise() += (b0 * eBitsSeq0.transpose()).row(0);
//     eEBatch.colwise() += (b1 * eBitsSeq1.transpose()).transpose().col(0);
    
//     /* FIXME: Parallelize */
//     for (int idx1 = 0; idx1 < nBatch1; ++idx1) {
//         for (int idx0 = 0; idx0 < nBatch0; ++idx0) {
//             real Etmp = eEBatch(idx1, idx0);
//             if (Etmp > Emin) {
//                 continue;
//             }
//             else if (Etmp == Emin) {
//                 xPairs->push_back(PackedBitSetPairArray::value_type(xBegin0 + idx0, xBegin1 + idx1));
//             }
//             else {
//                 Emin = Etmp;
//                 xPairs->clear();
//                 xPairs->push_back(PackedBitSetPairArray::value_type(xBegin0 + idx0, xBegin1 + idx1));
//             }
//         }
//     }
//     *E = Emin;
// }
    

