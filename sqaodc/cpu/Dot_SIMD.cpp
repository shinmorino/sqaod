#include "Dot_SIMD.h"

#ifdef __linux__
#include <x86intrin.h>
#endif

#ifdef _WIN32
#include <intrin.h>
#endif

namespace sqaod_cpu {

#ifdef __SSE2__

// #define USE_PREFETCH

double dot_sse2(const double *v0, const double *v1, sq::SizeType N) {
#ifdef USE_PREFETCH
    enum {
        PrefetchDistance = 32,
    };
#endif
    __m128d sum2 = { 0., 0. };
    int nCacheLines = sq::divru(N, 8);
#ifdef USE_PREFETCH
    for (int idx = 0; idx < PrefetchDistance; idx += 8) {
        _mm_prefetch(v0 + idx, _MM_HINT_T0);
        _mm_prefetch(v1 + idx, _MM_HINT_T0);
    }
#endif
    for (int idx = 0; idx < nCacheLines; ++idx) {
#ifdef USE_PREFETCH
        _mm_prefetch(v0 + PrefetchDistance, _MM_HINT_T0);
        _mm_prefetch(v1 + PrefetchDistance, _MM_HINT_T0);
#endif
        __m128d v0_2, v1_2;
        v0_2 = _mm_load_pd(v0); v1_2 = _mm_load_pd(v1);
        __m128d prod0 = _mm_mul_pd(v0_2, v1_2);
        v0_2 = _mm_load_pd(v0 + 2); v1_2 = _mm_load_pd(v1 + 2);
        __m128d prod1 = _mm_mul_pd(v0_2, v1_2);
        v0_2 = _mm_load_pd(v0 + 4); v1_2 = _mm_load_pd(v1 + 4);
        __m128d prod2 = _mm_mul_pd(v0_2, v1_2);
        v0_2 = _mm_load_pd(v0 + 6); v1_2 = _mm_load_pd(v1 + 6);
        __m128d prod3 = _mm_mul_pd(v0_2, v1_2);

        __m128d sum2_01 = _mm_add_pd(prod0, prod1);
        __m128d sum2_23 = _mm_add_pd(prod2, prod3);
        __m128d sum2_0123 = _mm_add_pd(sum2_01, sum2_23);
        sum2 = _mm_add_pd(sum2, sum2_0123);

        v0 += 8; v1 += 8;
    }
    const __m128d sum2lo = _mm_shuffle_pd(sum2, sum2, 0x1);
    const __m128d sum = _mm_add_pd(sum2, sum2lo);
    return _mm_cvtsd_f64(sum);
}



float dot_sse2(const float *v0, const float *v1, sq::SizeType N) {
#ifdef USE_PREFETCH
    enum {
        PrefetchDistance = 64
    };
#endif

    __m128 sum4 = { 0.f, 0.f, 0.f, 0.f };
    int nCacheLines = sq::divru(N, 16);
#ifdef USE_PREFETCH
    for (int idx = 0; idx < PrefetchDistance; idx += 16) {
        _mm_prefetch(v0 + idx, _MM_HINT_T0);
        _mm_prefetch(v1 + idx, _MM_HINT_T0);
    }
#endif
    for (int idx = 0; idx < nCacheLines; ++idx) {
#ifdef USE_PREFETCH
        _mm_prefetch(v0 + PrefetchDistance, _MM_HINT_T0);
        _mm_prefetch(v1 + PrefetchDistance, _MM_HINT_T0);
#endif
        __m128 v0_2, v1_2;
        v0_2 = _mm_load_ps(v0); v1_2 = _mm_load_ps(v1);
        __m128 prod0 = _mm_mul_ps(v0_2, v1_2);
        v0_2 = _mm_load_ps(v0 + 4); v1_2 = _mm_load_ps(v1 + 4);
        __m128 prod1 = _mm_mul_ps(v0_2, v1_2);
        v0_2 = _mm_load_ps(v0 + 8); v1_2 = _mm_load_ps(v1 + 8);
        __m128 prod2 = _mm_mul_ps(v0_2, v1_2);
        v0_2 = _mm_load_ps(v0 + 12); v1_2 = _mm_load_ps(v1 + 12);
        __m128 prod3 = _mm_mul_ps(v0_2, v1_2);
        __m128 sum01 = _mm_add_ps(prod0, prod1);
        __m128 sum23 = _mm_add_ps(prod2, prod3);
        __m128 sum0123 = _mm_add_ps(sum01, sum23);
        sum4 = _mm_add_ps(sum4, sum0123);
        
        v0 += 16; v1 += 16;
    }
    // https://qiita.com/beru/items/fff00c19968685dada68
    // loDual = ( -, -, x1, x0 )
    const __m128 loDual = sum4;
    // hiDual = ( -, -, x3, x2 )
    const __m128 hiDual = _mm_movehl_ps(sum4, sum4);
    // sumDual = ( -, -, x1+x3, x0+x2 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0+x2 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1+x3 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0+x1+x2+x3 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

#endif


#ifdef __AVX2__

double dot_avx2(const double *v0, const double *v1, sq::SizeType N) {
#ifdef USE_PREFETCH
    enum {
        PrefetchDistance = 64,
    };
#endif
    __m256d sum4_0 = { 0. };
    __m256d sum4_1 = { 0. };
    int nCacheLines = sq::divru(N, 8);
#ifdef USE_PREFETCH
    for (int idx = 0; idx < PrefetchDistance; idx += 8) {
        _mm_prefetch(v0 + idx, _MM_HINT_T0);
        _mm_prefetch(v1 + idx, _MM_HINT_T0);
    }
#endif
    for (int idx = 0; idx < nCacheLines; ++idx) {
#ifdef USE_PREFETCH
        _mm_prefetch(v0 + PrefetchDistance, _MM_HINT_T0);
        _mm_prefetch(v1 + PrefetchDistance, _MM_HINT_T0);
#endif
        __m256d v0_4, v1_4;
        v0_4 = _mm256_load_pd(v0); v1_4 = _mm256_load_pd(v1);
        sum4_0 = _mm256_fmadd_pd(v0_4, v1_4, sum4_0);
        v0_4 = _mm256_load_pd(v0 + 4); v1_4 = _mm256_load_pd(v1 + 4);
        sum4_1 = _mm256_fmadd_pd(v0_4, v1_4, sum4_1);

        v0 += 8; v1 += 8;
    }
    __m256d sum4 = _mm256_add_pd(sum4_0, sum4_1);
    // hiDual = ( x3, x2 )
    const __m128d hiDual = _mm256_extractf128_pd(sum4, 1);
    // loDual = ( x1, x0 )
    const __m128d loDual = _mm256_castpd256_pd128(sum4);
    // sumDual = ( x3 + x1, x2 + x0 )
    const __m128d sumDual = _mm_add_pd(hiDual, loDual);
    // sumDual = ( -        x3 + x1 )
    const __m128d sumDualLo = _mm_shuffle_pd(sumDual, sumDual, 0x1);
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128d sum = _mm_add_pd(sumDual, sumDualLo);
    return _mm_cvtsd_f64(sum);
}


float dot_avx2(const float *v0, const float *v1, sq::SizeType N) {
#ifdef USE_PREFETCH
    enum {
        PrefetchDistance = 96
    };
#endif
    __m256 sum8_0 = { 0.f };
    __m256 sum8_1 = { 0.f };
    int nCacheLines = sq::divru(N, 16);
#ifdef USE_PREFETCH
    for (int idx = 0; idx < PrefetchDistance; idx += 16) {
        _mm_prefetch(v0 + idx, _MM_HINT_T0);
        _mm_prefetch(v1 + idx, _MM_HINT_T0);
    }
#endif
    for (int idx = 0; idx < nCacheLines; ++idx) {
#ifdef USE_PREFETCH
        _mm_prefetch(v0 + PrefetchDistance, _MM_HINT_T0);
        _mm_prefetch(v1 + PrefetchDistance, _MM_HINT_T0);
#endif
        __m256 v0_8, v1_8;
        v0_8 = _mm256_load_ps(v0); v1_8 = _mm256_load_ps(v1);
        sum8_0 = _mm256_fmadd_ps(v0_8, v1_8, sum8_0);
        v0_8 = _mm256_load_ps(v0 + 8); v1_8 = _mm256_load_ps(v1 + 8);
        sum8_1 = _mm256_fmadd_ps(v0_8, v1_8, sum8_1);

	v0 += 16; v1 += 16;
    }
    __m256 sum8 = _mm256_add_ps(sum8_0, sum8_1);

    /* https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally */
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(sum8, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(sum8);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

#endif


double dot_naive(const double *v0, const double *v1, sq::SizeType N) {
    double sum = 0.;
    for (sq::IdxType idx = 0; idx < N; ++idx)
        sum += v0[idx] * v1[idx];
    return sum;
}

float dot_naive(const float *v0, const float *v1, sq::SizeType N) {
    float sum = 0.;
    for (sq::IdxType idx = 0; idx < N; ++idx)
        sum += v0[idx] * v1[idx];
    return sum;
}

}
