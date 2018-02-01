#pragma once

#include <common/types.h>


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
void multiply(V *values, SizeType size, const V v) {
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        values[idx] *= v;
}

template<class newV, class V> inline
void x_from_q(newV *dst, const V *src, SizeType size) {
    for (IdxType idx = 0; idx < (IdxType)size; ++idx)
        dst[idx] = (newV)((src[idx] + 1) / 2);
}

}
