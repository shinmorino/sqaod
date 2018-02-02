#pragma once

#include <common/types.h>
#include <limits>
#include <algorithm>


namespace sqaod {

template<class newV, class V> inline
void fill(newV *dst, const V &src, SizeType size) {
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        dst[idx] = (newV)src;
}

template<class newV, class V> inline
void cast(newV *dst, const V *src, SizeType size) {
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        dst[idx] = (newV)src[idx];
}

template<class V> inline
void multiply(V *values, const V &v, SizeType size) {
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        values[idx] *= v;
}

template<class V> inline
V sum(const V *values, SizeType size) {
    V v = V(0.);
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        v += values[idx];
    return v;
}

template<class V> inline
V min(V *values, SizeType size) {
    V v = std::numeric_limits<V>::max();
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        v = std::min(v, values[idx]);
    return v;
}

template<class newV, class V> inline
void x_from_q(newV *dst, const V *src, SizeType size) {
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        dst[idx] = (newV)((src[idx] + 1) / 2);
}


/* specialization */
double sum(const double *values, SizeType size);
float sum(const float *values, SizeType size);

}
