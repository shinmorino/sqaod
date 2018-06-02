#pragma once

#include <sqaodc/common/Common.h>

namespace sqaod_cpu {

namespace sq = sqaod;

#ifdef __AVX2__

double dot_avx2(const double *v0, const double *v1, sq::SizeType N);

float dot_avx2(const float *v0, const float *v1, sq::SizeType N);

#endif

#ifdef __SSE2__

double dot_sse2(const double *v0, const double *v1, sq::SizeType N);

float dot_sse2(const float *v0, const float *v1, sq::SizeType N);

#endif


double dot_naive(const double *v0, const double *v1, sq::SizeType N);

float dot_naive(const float *v0, const float *v1, sq::SizeType N);


}
