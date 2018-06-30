#pragma once

#include <common/types.h>
#include <iterator>
#include <algorithm>

namespace sqaod_cuda {

namespace sq = sqaod;

template<template<class> class OpType, class real>
struct OpOutPtr {
    typedef real value_type;
    typedef OpType<real> Op;

    explicit
    OpOutPtr(real *_d_data, sq::SizeType _stride, Op _op) : d_data(_d_data), stride(_stride), op(_op) { }

    void addYOffset(sq::IdxType yOffset) {
        d_data = &d_data[yOffset * stride];
    }

    __device__ __forceinline__
    Op operator*() const {
        return op(*d_data);
    }
    __device__ __forceinline__
    Op operator[](sq::SizeType idx) const {
        return op(d_data[idx]);
    }
    __device__ __forceinline__
    Op operator()(sq::SizeType x, sq::SizeType y) const {
        return op(d_data[x + y * stride]);
    }

    real *d_data;
    sq::SizeType stride;
    Op op;
};


template<class real>
struct NullOutOp {
    explicit NullOutOp() { }
    explicit __device__ __forceinline__
    NullOutOp(real &_d_value) : d_value(&_d_value) { }

    __device__ __forceinline__
    NullOutOp operator()(real &_d_value) const {
        return NullOutOp(_d_value);
    }
    __device__ __forceinline__
    real operator=(const real &v) const {
        return *d_value = v;
    }
    real *d_value;
};

template<class real>
struct AddAssignOutOp {
    explicit AddAssignOutOp(real _addAssignFactor, real _alpha) : addAssignFactor(_addAssignFactor), alpha(_alpha) { }
    explicit __device__ AddAssignOutOp(real &_d_data, const AddAssignOutOp &op) : d_data(&_d_data), addAssignFactor(op.addAssignFactor), alpha(op.alpha) { }

    __device__ __forceinline__
    AddAssignOutOp operator()(real &_d_value) const {
        return AddAssignOutOp<real>(_d_value, *this);
    }
    __device__ __forceinline__
    real operator=(const real &v) const {
        return *d_data = addAssignFactor * *d_data + alpha * v;
    }

    real *d_data;
    real addAssignFactor;
    real alpha;
};

template<class real>
struct MulOutOp {
    explicit MulOutOp(real _alpha) : alpha(_alpha) { }
    explicit __device__ MulOutOp(real &_d_data, const MulOutOp &op) : d_data(&_d_data), alpha(op.alpha) { }

    __device__ __forceinline__
    MulOutOp operator()(real &_d_data) const {
        return MulOutOp(_d_data, *this);
    }
    __device__ __forceinline__
    real operator=(const real &v) const {
        return *d_data = alpha * v;
    }

    real *d_data;
    real alpha;
};


template<class real>
OpOutPtr<NullOutOp, real> NullOutPtr(real *d_data, sq::SizeType stride = 0) {
    return OpOutPtr<NullOutOp, real>(d_data, stride, NullOutOp<real>());
}

template<class real>
OpOutPtr<AddAssignOutOp, real> AddAssignOutPtr(real *d_data, real _mulFactor, real _alpha, sq::SizeType stride = 0) {
    return OpOutPtr<AddAssignOutOp, real>(d_data, stride, AddAssignOutOp<real>(_mulFactor, _alpha));
}

template<class real>
OpOutPtr<MulOutOp, real> MulOutPtr(real *d_data, real _mulFactor, sq::SizeType stride = 0) {
    return OpOutPtr<MulOutOp, real>(d_data, stride, MulOutOp<real>(_mulFactor));
}

/* Input iterators */
template<class real>
struct InPtr {
    typedef real value_type;

    explicit
    InPtr(const real *_d_data, sq::SizeType _stride = 0) : d_data(_d_data), stride(_stride) { }

    void addYOffset(sq::IdxType yOffset) {
        d_data = &d_data[yOffset * stride];
    }

    __device__ __forceinline__
    real operator*() const {
        return *d_data;
    }
    __device__ __forceinline__
    real operator[](sq::SizeType idx) const {
        return d_data[idx];
    }
    __device__ __forceinline__
    real operator()(sq::SizeType x, sq::SizeType y) const {
        return d_data[x + y * stride];
    }

    const real *d_data;
    sq::SizeType stride;
};

template<class real>
struct InScalarPtr {
    typedef real value_type;

    explicit
    InScalarPtr(const real *_d_data) : d_data(_d_data) { }

    void addYOffset(sq::IdxType yOffset) {  }

    __device__ __forceinline__
    real operator*() const {
        return *d_data;
    }
    __device__ __forceinline__
    real operator[](sq::SizeType idx) const {
        return *d_data;
    }
    __device__ __forceinline__
    real operator()(sq::SizeType x, sq::SizeType y) const {
        return *d_data;
    }

    const real *d_data;
};

template<class real>
struct InConstPtr {
    typedef real value_type;

    explicit
    InConstPtr(const real &_v) : v(_v) { }

    void addYOffset(sq::IdxType yOffset) {  }

    __device__ __forceinline__
    real operator*() const {
        return v;
    }
    __device__ __forceinline__
    real operator[](sq::SizeType idx) const {
        return v;
    }
    __device__ __forceinline__
    real operator()(sq::SizeType x, sq::SizeType y) const {
        return v;
    }
    real v;
};

template<class real>
struct InRowBroadcastPtr {
    typedef real value_type;

    explicit
    InRowBroadcastPtr(const real *_d_data) : d_data(_d_data) { }

    void addYOffset(sq::IdxType yOffset) { }

    __device__ __forceinline__
    real operator[](sq::SizeType idx) const {
        return d_data[idx];
    }

    __device__ __forceinline__
    real operator()(sq::SizeType x, sq::SizeType y) const {
        return d_data[x];
    }

    const real *d_data;
};

template<class real>
struct InColumnBroadcastPtr {
    typedef real value_type;

    explicit
    InColumnBroadcastPtr(const real *_d_data) : d_data(_d_data) { }

    void addYOffset(sq::IdxType yOffset) {
        d_data += yOffset;
    }

    __device__ __forceinline__
    real operator[](sq::SizeType idx) const {
        return d_data[idx];
    }

    __device__ __forceinline__
    real operator()(sq::SizeType x, sq::SizeType y) const {
        return d_data[y];
    }

    const real *d_data;
};

template<class real>
struct InDiagonalPtr {
    typedef real value_type;
    typedef InDiagonalPtr<real> SelfType;

    explicit __host__ __device__ __forceinline__
    InDiagonalPtr(const real *_d_data, sq::SizeType _stride, sq::IdxType _xOffset, sq::IdxType _yOffset)
        : d_data(_d_data), stride(_stride), xOffset(_xOffset), yOffset(_yOffset) { }

    __device__ __forceinline__
    real operator[](sq::SizeType idx) const {
        int x = idx + xOffset;
        int y = idx + yOffset;
        return d_data[x + y * stride];
    }

    __device__ __forceinline__
    SelfType operator+(sq::IdxType v) const {
        return SelfType(d_data, stride, xOffset + v, yOffset + v);
    }

    const real *d_data;
    int stride, xOffset, yOffset;
};

/* iterators for specific use cases*/

template<class real>
struct InPtrWithInterval {
    typedef real value_type;
    typedef InPtrWithInterval SelfType;
    __host__ __device__
    InPtrWithInterval(const real *_d_data, sq::SizeType _stride, sq::IdxType _offset) : d_data(_d_data), stride(_stride), offset(_offset) { }
    __device__ __forceinline__
    const real &operator[](sq::SizeType idx) const {
        return d_data[offset + idx * stride];
    }

    const real *d_data;
    sq::SizeType stride;
    sq::IdxType offset;
};


template<class real>
struct InLinear2dPtr {
    typedef real value_type;
    typedef InLinear2dPtr SelfType;
    __host__ __device__
    InLinear2dPtr(const real *_d_data,
                  sq::SizeType _stride, sq::IdxType _width, sq::IdxType _offset = 0) :
                  d_data(_d_data), stride(_stride), width(_width), offset(_offset) { }
    __device__ __forceinline__
    const real &operator[](sq::SizeType idx) const {
        int x = (idx + offset) % width;
        int y = (idx + offset) / width;
        return d_data[x + y * stride];
    }
    __device__ __forceinline__
    SelfType operator+(sq::IdxType v) const {
        return SelfType(d_data, stride, width, offset + v);
    }

    const real *d_data;
    sq::SizeType stride;
    sq::SizeType width;
    sq::SizeType offset;
};


/* Functors for offsets */
struct Linear {
    Linear(sq::IdxType _a) : a(_a) { }
    __device__
    sq::IdxType operator[](sq::IdxType idx) const { return a * idx; }
    sq::SizeType a;
};


/* base traits class for CUB iteratos */
template<class V>
struct base_iterator_traits {
    using difference_type   = ptrdiff_t;
    typedef V                 value_type;
    using pointer           = V*;
    using reference         = V&;
    using iterator_category = std::random_access_iterator_tag;
};

}


namespace std {

template<template<class> class OpType, class real>
struct iterator_traits<sqaod_cuda::OpOutPtr<OpType, real> > : sqaod_cuda::base_iterator_traits<real> { };

template<class real>
struct iterator_traits<sqaod_cuda::InDiagonalPtr<real>> : sqaod_cuda::base_iterator_traits<real> { };
template<class real>
struct iterator_traits<sqaod_cuda::InLinear2dPtr<real>> : sqaod_cuda::base_iterator_traits<real> { };

template<>
struct iterator_traits<sqaod_cuda::Linear> : sqaod_cuda::base_iterator_traits<int> { };

}

#include "cudafuncs.h"

namespace sqaod_cuda {

namespace sq = sqaod;

template<class Op>  __global__
void transformBlock2dKernel(Op op, sq::IdxType blockDimYOffset) {
    dim3 blockIdxWithOffset(blockIdx);
    blockIdxWithOffset.y += blockDimYOffset;
    op(blockDim, blockIdxWithOffset, threadIdx);
}

template<class Op>
void transformBlock2d(const Op &op, sq::SizeType nBlocksX, sq::SizeType nBlocksY, const dim3 &blockDim, cudaStream_t stream) {
    sq::SizeType blockIdxYStep = 65535 / blockDim.y;
    for (sq::IdxType blockIdxYOffset = 0; blockIdxYOffset < nBlocksY; blockIdxYOffset += blockIdxYStep) {
        int blockDimYSpan = std::min(nBlocksY - blockIdxYOffset, blockIdxYStep);
        dim3 gridDim(nBlocksX, blockDimYSpan);
        transformBlock2dKernel<<<gridDim, blockDim, 0, stream>>>(op, blockIdxYOffset);
        DEBUG_SYNC;
    }
}


template<class Op>  __global__
void transform2dKernel(Op op, sq::SizeType width, sq::SizeType height, sq::IdxType offset) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y + offset;
    if ((gidx < width) && (gidy < height))
        op(gidx, gidy);
}

template<class Op>
void transform2d(const Op &op, sq::SizeType width, sq::SizeType height, const dim3 &blockDim, cudaStream_t stream) {
    sq::SizeType yStep = (65535 / blockDim.y) * blockDim.y;
    for (sq::IdxType idx = 0; idx < height; idx += yStep) {
        int hSpan = std::min(height - idx, yStep);
        dim3 gridDim(divru(width, blockDim.x), divru(hSpan, blockDim.y));
        transform2dKernel<<<gridDim, blockDim, 0, stream>>>(op, width, height, idx);
        DEBUG_SYNC;
    }
}

template<class OutType, class InType>
void transform2d(OutType d_out, InType d_in, sq::SizeType width, sq::SizeType height, const dim3 &blockDim, cudaStream_t stream) {
    auto op = [=]__device__(int gidx, int gidy) {
        d_out(gidx, gidy) = (typename OutType::value_type)d_in(gidx, gidy);
    };
    transform2d(op, width, height, blockDim, stream);
}

}
