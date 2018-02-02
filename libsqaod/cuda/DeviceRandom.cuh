#pragma once

#include <common/defines.h>
#include <cuda_runtime.h>
#include <curand_mtgp32.h>

namespace sqaod_cuda {

    
enum {
    randGenSize = CURAND_NUM_MTGP32_PARAMS * THREAD_NUM
};


void deviceRandomMakeKernelState(curandStateMtgp32_t *d_randStates_,
                                 mtgp32_kernel_params_t *kernelParams,
                                 unsigned long long seed, cudaStream_t stream);

void deviceGenRand(int *d_buffer, int begin, int end, int nToGenerate, int bufSize,
                   curandStateMtgp32_t *d_randStates, cudaStream_t stream);



}


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

