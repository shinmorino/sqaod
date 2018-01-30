#include "DeviceRandomKernel.cuh"
#include <curand_mtgp32_host.h>
#include <curand_mtgp32_kernel.h>
#include "cudafuncs.h"
#include <device_launch_parameters.h>

void sqaod_cuda::
deviceRandomMakeKernelState(curandStateMtgp32_t *d_randStates_,
                            mtgp32_kernel_params_t *d_kernelParams,
                            unsigned long long seed, cudaStream_t stream) {
    throwOnError(curandMakeMTGP32KernelState(d_randStates_,
                                             MTGPDC_PARAM_TABLE,
                                             d_kernelParams, CURAND_NUM_MTGP32_PARAMS, seed));
}


__global__
static void genRandKernel(int *d_buffer, int offset, int nNums, int bufLen,
                          curandStateMtgp32_t *d_state) {
    /* bufLen must be 2^n */
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    offset = (offset + gid) % bufLen;
    for (int idx = 0; idx < nNums; idx += sqaod_cuda::randGenSize) {
        int r = curand(&d_state[blockIdx.x]);
        d_buffer[offset] = r;
        offset = (offset + sqaod_cuda::randGenSize) % bufLen;
    }
}


void sqaod_cuda::
deviceGenRand(int *d_buffer, int begin, int end, int nToGenerate, int bufSize,
              curandStateMtgp32_t *d_randStates, cudaStream_t stream) {
    genRandKernel<<<CURAND_NUM_MTGP32_PARAMS, THREAD_NUM, 0, stream>>>
            (d_buffer, end, nToGenerate, bufSize, d_randStates);
    DEBUG_SYNC;
}
