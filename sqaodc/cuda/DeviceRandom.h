/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/defines.h>
#include <sqaodc/cuda/Device.h>
#include <cuda_runtime.h>
#include <curand_mtgp32.h>

namespace sqaod_cuda {

namespace sq = sqaod;

class DeviceRandom {
public:
    DeviceRandom();
    DeviceRandom(Device &device, DeviceStream *devStream = NULL);
    ~DeviceRandom();

    void assignDevice(Device &device, DeviceStream *devStreram = NULL);

    void deallocate();
    
    void setRequiredSize(sq::SizeType requiredSize);

    void seed();

    void seed(unsigned int seed);

    sq::SizeType getNRands() const;

    void generate();

    const unsigned int *get(sq::SizeType nRands, sq::IdxType *offset, sq::SizeType *posToWrap,
                            int alignment = 1);

    void synchronize();

private:
    void deallocateStates();
    void deallocateBuffer();

    void deviceRandomMakeKernelState(curandStateMtgp32_t *d_randStates_,
                                     mtgp32_kernel_params_t *kernelParams,
                                     unsigned long long seed, cudaStream_t stream);

    void deviceGenRand(int *d_buffer, int begin, int end, int nToGenerate, int bufSize,
                       curandStateMtgp32_t *d_randStates, cudaStream_t stream);


    DeviceObjectAllocator *devAlloc_;
    cudaStream_t stream_;
    
    sq::SizeType requiredSize_;
    sq::SizeType internalBufSize_;
    
    curandStateMtgp32_t *d_randStates_;
    mtgp32_kernel_params_t *d_kernelParams_;
    unsigned int *d_buffer_;
    int begin_;
    int end_;
};

}
