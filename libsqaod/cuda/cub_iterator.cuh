#pragma once

#include <common/types.h>
#include <iterator>

namespace sq = sqaod;

namespace sqaod_cuda {

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
struct StridedInPtr {
    typedef real value_type;
    typedef StridedInPtr SelfType;
    __host__ __device__
    StridedInPtr(const real *_d_data, sq::SizeType _stride, sq::IdxType _offset) : d_data(_d_data), stride(_stride), offset(_offset) { }
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
struct InDotPtr {
    typedef InDotPtr<real> SelfType;

    __host__ __device__
    InDotPtr(const real *_d_x, const real *_d_y) : d_x(_d_x), d_y(_d_y) { }
    __device__ __forceinline__
    real operator[](sq::IdxType idx) const {
        return d_x[idx] * d_y[idx];
    }

    __device__ __forceinline__
    real operator[](const int2 &idx2) const {
        return d_x[idx2.x] * d_y[idx2.y];
    }

    __device__ __forceinline__
    SelfType operator+(sq::IdxType idx) const {
        return SelfType(&d_x[idx], &d_y[idx]);
    }

    const real *d_x, *d_y;
};

/* Functors for offsets */
struct Linear {
    Linear(sq::IdxType _a, sq::IdxType _b) : a(_a), b(_b) { }
    __device__
    sq::IdxType operator[](sq::IdxType idx) const { return a * idx + b; }
    sq::IdxType a, b;
};


struct Offset2way  {
    __host__
    Offset2way(const sq::IdxType *_d_offset, sq::SizeType _segLen)
            : d_offset(_d_offset), segLen(_segLen) { }

    __device__ __forceinline__
    int2 operator[](sq::IdxType idx) const {
        return make_int2(idx * segLen, d_offset[idx] * segLen);
    }
    const sq::IdxType *d_offset;
    sq::SizeType segLen;
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
struct iterator_traits<sqaod_cuda::StridedInPtr<real>> : sqaod_cuda::base_iterator_traits<real> { };

template<class real>
struct iterator_traits<sqaod_cuda::InDotPtr<real>> : sqaod_cuda::base_iterator_traits<real> { };

template<>
struct iterator_traits<sqaod_cuda::Offset2way> : sqaod_cuda::base_iterator_traits<int2> { };

template<>
struct iterator_traits<sqaod_cuda::Linear> : sqaod_cuda::base_iterator_traits<int> { };

}
