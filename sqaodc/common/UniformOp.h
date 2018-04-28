#pragma once

#include <sqaodc/common/types.h>
#include <limits>
#include <algorithm>


namespace sqaod {

template<class newV, class V> inline
void fill(newV *dst, const V &src, SizeType cols, SizeType rows = 1, SizeType stride = -1) {
    for (IdxType row = 0; row < rows; ++row) {
        newV *dstrow = &dst[stride * row];
        for (IdxType idx = 0; idx < (IdxType)cols; ++idx)
            dstrow[idx] = (newV)src;
    }
}

template<class newV, class V> inline
void cast(newV *dst, const V *src, SizeType cols, SizeType rows = 1, SizeType dstStride = -1, SizeType srcStride = -1) {
    for (IdxType row = 0; row < rows; ++row) {
        newV *dstrow = &dst[dstStride * row];
        const V *srcrow = &src[srcStride * row];
        for (IdxType idx = 0; idx < cols; ++idx)
            dstrow[idx] = (newV)srcrow[idx];
    }
}

template<class V> inline
void multiply(V *values, const V &v, SizeType cols, SizeType rows = 1, SizeType stride = -1) {
    for (IdxType row = 0; row < rows; ++row) {
        V *vrow = &values[stride * row];
        for (IdxType idx = 0; idx < cols; ++idx)
            vrow[idx] *= v;
    }
}

template<class V> inline
V min(V *values, SizeType cols, SizeType rows = 1, SizeType stride = -1) {
    V v = std::numeric_limits<V>::max();
    for (IdxType row = 0; row < rows; ++row) {
        const V *vrow = &values[stride * row];
        for (IdxType idx = 0; idx < cols; ++idx)
            v = std::min(v, vrow[idx]);
    }
    return v;
}

template<class newV, class V> inline
void x_from_q(newV *dst, const V *src, SizeType cols, SizeType rows = 1,
              SizeType dstStride = -1, SizeType srcStride = -1) {
    for (IdxType row = 0; row < rows; ++row) {
        newV *dstrow = &dst[dstStride * row];
        const V *srcrow = &src[srcStride * row];
        for (IdxType idx = 0; idx < cols; ++idx)
            dstrow[idx] = (newV)((srcrow[idx] + 1) / 2);
    }
}

template<class newV, class V> inline
void x_to_q(newV *dst, const V *src, SizeType cols, SizeType rows = 0,
            SizeType dstStride = -1, SizeType srcStride = -1) {
    for (IdxType row = 0; row < rows; ++row) {
        newV *dstrow = &dst[dstStride * row];
        const V *srcrow = &src[srcStride * row];
        for (IdxType idx = 0; idx < cols; ++idx)
            dstrow[idx] = (newV)((srcrow[idx] * 2) - 1);
    }
}

/* specialization */
double sum(const double *values, SizeType cols, SizeType rows = 0, SizeType stride = -1);
float sum(const float *values, SizeType cols, SizeType rows = 0, SizeType stride = -1);

}
