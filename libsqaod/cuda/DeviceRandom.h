/* -*- c++ -*- */
#ifndef DEVICERANDOM_H__
#define DEVICERANDOM_H__

#include <cuda_runtime.h>
#include <curand_mtgp32.h>


namespace sqaod_cuda {

class DeviceRandom {
public:
    DeviceRandom();
    ~DeviceRandom();

    enum {
        bufLen = 16 * 1024 * 1024,
        randsGenSize = 256 * 20
    };
    
    void allocate(int nNums);

    void deallocate();
    
    void setSeed(unsigned long long seed);
    
    void reset();

    int getNRands() const;
    
    void generate(cudaStream_t stream = NULL);
    
    const int *get(int nRands, int *offset, cudaStream_t stream = NULL);
    
private:
    int bufLen_;
    
    curandStateMtgp32_t *d_randStates_;
    mtgp32_kernel_params_t *d_kernelParams_;
    int *d_buffer_;
    int begin_;
    int end_;
};

}

#endif
