#include "DeviceRandom.cuh"
#include <curand_mtg32_host.h>
#include <curand_mtg32_kernel.h>


void deviceRandomMakeKernelState(curandStateMtgp32_t *d_randStates_,
                                 mtgp32_kernel_params_t *kernelParams,
                                 unsigned long long seed, cudaStream_t stream) {
    throwOnError(curandMakeMTGP32KernelState(d_randStates_,
                                             MTGPDC_PARAM_TABLE,
                                             d_kernelParams_, 200, seed));
}


__global__
static void genRandKernel(int *d_buffer, int offset, int nNums, int bufLen,
                          curandStateMtgp32_t *d_state) {
    /* bufLen must be 2^n */
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    offset = (offset + gid) % bufLen;
    for (int idx = 0; idx < nNums; idx += 20 * 256) {
        int r = curand(d_state);
        curand_mtgp32_specific(&d_state[blockIdx.x], threadIdx.x, 256);
        offset = (offset + 20 * 256) % bufLen;
    }
}


void sqaod_cuda::
deviceGenRand(int *d_buffer, int begin, int end, int nToGenerate, int bufSize,
              curandStateMtgp32_t *d_randStates, cudaStream_t stream) {
    genRandKernel<<<CURAND_NUM_MTGP32_PARAMS, THREAD_NUM, 0, stream>>>
            (d_buffer, end, nToGenerate, bufSize, d_randStates);
    DEBUG_SYNC;
}
