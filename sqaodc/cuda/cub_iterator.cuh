#pragma once

#include <common/types.h>
#include <iterator>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
struct AddAssign {
    __device__ AddAssign(real &_d_value, real _mulFactor, real _alpha) : d_value(_d_value), mulFactor(_mulFactor), alpha(_alpha) { }
    __device__ __forceinline__
    real operator=(const real &v) const {
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
    __device__ __forceinline__
    Ref operator*() const {
        return Ref(*d_data, mulFactor, alpha);
    }
    __device__ __forceinline__
    Ref operator[](sq::SizeType idx) const {
        return Ref(d_data[idx], mulFactor, alpha);
    }

    real *d_data;
    real mulFactor;
    real alpha;
};


template<class real>
struct Mul{
    __device__ Mul(real &_d_value, real _alpha) : d_value(_d_value), alpha(_alpha) { }
    __device__ __forceinline__
    real operator=(const real &v) const {
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
    __device__ __forceinline__
    Ref operator*() const {
        return Ref(*d_data, alpha);
    }
    __device__ __forceinline__
    Ref operator[](sq::SizeType idx) const {
        return Ref(d_data[idx], alpha);
    }

    real *d_data;
    real alpha;
};



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
    __device__ __forceinline__
    SelfType operator+(sq::IdxType v) const {
        return SelfType(d_data + v, stride, offset);
    }

    const real *d_data;
    sq::SizeType stride;
    sq::IdxType offset;
};


template<class real>
struct In2dPtr {
    typedef real value_type;
    typedef In2dPtr SelfType;
    __host__ __device__
    In2dPtr(const real *_d_data,
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


template<class real>
struct iterator_traits<sqaod_cuda::AddAssignDevPtr<real> > : sqaod_cuda::base_iterator_traits<real> { };
template<class real>
struct iterator_traits<sqaod_cuda::MulOutDevPtr<real> > : sqaod_cuda::base_iterator_traits<real> { };
template<class real>
struct iterator_traits<sqaod_cuda::InPtrWithInterval<real>> : sqaod_cuda::base_iterator_traits<real> { };
template<class real>
struct iterator_traits<sqaod_cuda::In2dPtr<real>> : sqaod_cuda::base_iterator_traits<real> { };

template<class Vout, class Vin0, class Vin1>
struct iterator_traits<sqaod_cuda::In2TypeDotPtr<Vout, Vin0, Vin1>> : sqaod_cuda::base_iterator_traits<Vout> { };

template<>
struct iterator_traits<sqaod_cuda::Offset2way> : sqaod_cuda::base_iterator_traits<int2> { };

template<>
struct iterator_traits<sqaod_cuda::Linear> : sqaod_cuda::base_iterator_traits<int> { };

}
