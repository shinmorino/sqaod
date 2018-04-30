#pragma once

#include <common/types.h>
#include <iterator>

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



template<class Vout, class Vin0, class Vin1>
struct In2TypeDotPtr {
    typedef In2TypeDotPtr<Vout, Vin0, Vin1> SelfType;

    __host__ __device__
    In2TypeDotPtr(const Vin0 *_d_x, const Vin1 *_d_y) : d_x(_d_x), d_y(_d_y) { }
    __device__ __forceinline__
    Vout operator[](sq::IdxType idx) const {
        return (Vout)d_x[idx] * (Vout)d_y[idx];
    }

    __device__ __forceinline__
    Vout operator[](const int2 &idx2) const {
        return (Vout)d_x[idx2.x] * (Vout)d_y[idx2.y];
    }

    __device__ __forceinline__
    SelfType operator+(sq::IdxType idx) const {
        return SelfType(&d_x[idx], &d_y[idx]);
    }

    const Vin0 *d_x;
    const Vin1 *d_y;
};

template<class real>
using InDotPtr = In2TypeDotPtr<real, real, real>;

/* Functors for offsets */
struct Linear {
    Linear(sq::IdxType _a, sq::IdxType _b) : a(_a), b(_b) { }
    __device__
    sq::IdxType operator[](sq::IdxType idx) const { return a * idx + b; }
    sq::IdxType a, b;
};


struct Offset2way  {
    __host__
    Offset2way(const sq::IdxType *_d_offset, sq::SizeType _stride0, sq::SizeType _stride1)
            : d_offset(_d_offset), stride0(_stride0), stride1(_stride1) { }

    __device__ __forceinline__
    int2 operator[](sq::IdxType idx) const {
        return make_int2(idx * stride0, d_offset[idx] * stride1);
    }
    const sq::IdxType *d_offset;
    sq::SizeType stride0, stride1;
};

__device__ __forceinline__
int2 operator+(const int2 &lhs, const int v) {
    return make_int2(lhs.x + v, lhs.y + v);
}


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

template<class Vout, class Vin0, class Vin1>
struct iterator_traits<sqaod_cuda::In2TypeDotPtr<Vout, Vin0, Vin1>> : sqaod_cuda::base_iterator_traits<Vout> { };

template<>
struct iterator_traits<sqaod_cuda::Offset2way> : sqaod_cuda::base_iterator_traits<int2> { };

template<>
struct iterator_traits<sqaod_cuda::Linear> : sqaod_cuda::base_iterator_traits<int> { };

}
