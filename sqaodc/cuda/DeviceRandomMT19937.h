/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/defines.h>
#include <sqaodc/cuda/Device.h>
#include <cuda_runtime.h>
#include <curand.h>

namespace sqaod_cuda {

namespace sq = sqaod;

class DeviceRandomMT19937 {
public:
    DeviceRandomMT19937();
    DeviceRandomMT19937(Device &device, DeviceStream *devStream = NULL);
    ~DeviceRandomMT19937();

    void assignDevice(Device &device, DeviceStream *devStreram = NULL);

    void deallocate();
    
    void setRequiredSize(sq::SizeType requiredSize);

    void seed();

    void seed(unsigned long long seed);

    sq::SizeType getNRands() const;

    void generate();

    const unsigned int *get(sq::SizeType nRands, sq::IdxType *offset, sq::SizeType *posToWrap,
                            int alignment = 1);

    void synchronize();

private:
    DeviceObjectAllocator *devAlloc_;
    cudaStream_t stream_;
    
    sq::SizeType requiredSize_;
    sq::SizeType internalBufSize_;

    curandGenerator_t gen_;
    
    unsigned int *d_buffer_[2];
    int activePlane_;
    int pos_;
};

}
