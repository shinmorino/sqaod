/* -*- c++ -*- */
#pragma once

#include <common/defines.h>
#include <cuda/Device.h>
#include <cuda_runtime.h>
#include <curand_mtgp32.h>

namespace sqaod_cuda {

class DeviceRandom {
public:
    DeviceRandom();
    DeviceRandom(Device &device, DeviceStream *devStream = NULL);
    ~DeviceRandom();

    void assignDevice(Device &device, DeviceStream *devStreram = NULL);

    void deallocate();

    void setRequiredSize(sqaod::SizeType requiredSize);

    void seed();
    
    void seed(unsigned long long seed);

    sqaod::SizeType getNRands() const;
    
    void generate();

    const int *get(sqaod::SizeType nRands, sqaod::IdxType *offset, sqaod::SizeType *posToWrap);
    
    void synchronize();

private:

void deviceRandomMakeKernelState(curandStateMtgp32_t *d_randStates_,
                                 mtgp32_kernel_params_t *kernelParams,
                                 unsigned long long seed, cudaStream_t stream);

void deviceGenRand(int *d_buffer, int begin, int end, int nToGenerate, int bufSize,
                   curandStateMtgp32_t *d_randStates, cudaStream_t stream);



    DeviceObjectAllocator *devAlloc_;
    cudaStream_t stream_;
    
    sqaod::SizeType requiredSize_;
    sqaod::SizeType internalBufSize_;
    
    curandStateMtgp32_t *d_randStates_;
    mtgp32_kernel_params_t *d_kernelParams_;
    int *d_buffer_;
    int begin_;
    int end_;
};

}
