#include "DeviceRandom.h"
#include "cudafuncs.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>

/* FIXME: use multiple states utilize more threads.
 *         use memory store */

using namespace sqaod_cuda;

static const int mega = 1024 * 1024;
    
DeviceRandom::DeviceRandom() {
    d_buffer_ = NULL;
}

DeviceRandom::~DeviceRandom() {
    if (d_buffer_ != NULL)
        deallocate();
}
    
void DeviceRandom::allocate(int nNums) {
    assert(d_buffer_ == NULL);
    bufLen_ = roundUp(nNums, mega) * 2;
    CUERR(cudaMalloc(&d_buffer_, sizeof(int) * bufLen_));
}
        

void DeviceRandom::deallocate() {
    cudaFree(d_buffer_);
    d_buffer_ = NULL;
}

void DeviceRandom::setSeed(unsigned long long seed) {
    if (d_randStates_ == NULL)
        throwOnError(cudaMalloc(&d_randStates_, sizeof(curandStateMtgp32_t) * 200));
    if (d_kernelParams_ == NULL)
        throwOnError(cudaMalloc(&d_kernelParams_, sizeof(mtgp32_kernel_params_t) * 200));
    throwOnError(curandMakeMTGP32KernelState(d_randStates_,
                                             MTGPDC_PARAM_TABLE,
                                             d_kernelParams_, 200, seed));
}
    
void DeviceRandom::reset() {
    begin_ = end_ = 0;
}


__global__
static void randGenKernel(int *d_buffer, int offset, int nNums, int bufLen,
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

int DeviceRandom::getNRands() const {
    return (end_ - begin_ + bufLen_) % bufLen_;
}
    

void DeviceRandom::generate(cudaStream_t stream) {
    /* mega must be a multiple of 51200 ( = 256 * 20 ) */
    int nToGenerate = bufLen - roundUp(getNRands(), (int)randsGenSize);
    randGenKernel<<<20, 256, 0, stream>>>(d_buffer_, end_, nToGenerate, bufLen_, d_randStates_);
    DEBUG_SYNC;
    end_ = (end_ + nToGenerate) % bufLen_;
}

const int *DeviceRandom::get(int nRands, int *offset, cudaStream_t stream) {
    if (getNRands() < nRands)
        generate(stream);
    assert(getNRands() < nRands);
      
    *offset = begin_;
    begin_ = (begin_ + nRands) % bufLen_;
    return d_buffer_;
}
