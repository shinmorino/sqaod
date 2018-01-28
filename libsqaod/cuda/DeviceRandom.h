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
    
    void allocate(sqaod::SizeType nNums);

    void deallocate();
    
    void setSeed(unsigned long long seed);

    sqaod::SizeType getNRands() const;
    
    void generate();
    
    const int *get(sqaod::SizeType nRands, sqaod::IdxType *offset);
    
private:
    DeviceObjectAllocator *devAlloc_;
    cudaStream_t stream_;
    
    int bufSize_;
    
    curandStateMtgp32_t *d_randStates_;
    mtgp32_kernel_params_t *d_kernelParams_;
    int *d_buffer_;
    int begin_;
    int end_;
};

}
