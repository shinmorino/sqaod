#pragma once

#include <cuda_runtime.h>


namespace {

/* [0, 1.) */
__device__ __forceinline__
float random(const int &v) {
    const float coef = 1.f/4294967296.f;
    return float(v) * coef; 
}

/* [0., 1.) in dobule */
__device__ __forceinline__
double random(const int2 &v) {
    unsigned long a= v.x >> 5, b = v.y >> 6; 
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0); 
}

}

