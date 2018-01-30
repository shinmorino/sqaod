#include "DeviceRandom.h"
#include "DeviceRandomKernel.cuh"
#include "cudafuncs.h"



/* FIXME: use multiple states utilize more threads.
 *         use memory store */

using namespace sqaod_cuda;

static const sqaod::SizeType mega = 1024 * 1024;
    
DeviceRandom::DeviceRandom(Device &device, DeviceStream *devStream) {
    assignDevice(device, devStream);
}

DeviceRandom::DeviceRandom() {
    requiredSize_ = -1;
    d_buffer_ = NULL;
    d_randStates_ = NULL;
    d_kernelParams_ = NULL;
    begin_ = end_ = 0;
}

DeviceRandom::~DeviceRandom() {
    if (d_buffer_ != NULL)
        deallocate();
}

void DeviceRandom::assignDevice(Device &device, DeviceStream *devStream) {
    devAlloc_ = device.objectAllocator();
    if (devStream == NULL)
        devStream = device.defaultStream();
    stream_ = devStream->getCudaStream();
}


void DeviceRandom::setRequiredSize(sqaod::SizeType requiredSize) {
    assert(d_buffer_ == NULL);
    requiredSize_ = requiredSize;
    /* Should give 2 chunks, 1 is for roundUp(), other is not to make size == 0 when filled up. */
    internalBufSize_ = roundUp(requiredSize_, (sqaod::SizeType)randGenSize)+ randGenSize * 2;
}
        

void DeviceRandom::deallocate() {
    assert(d_buffer_ != NULL);
    devAlloc_->deallocate(d_buffer_);
    devAlloc_->deallocate(d_randStates_);
    devAlloc_->deallocate(d_kernelParams_);

    d_buffer_ = NULL;
    d_randStates_ = NULL;
    d_kernelParams_ = NULL;
}

void DeviceRandom::seed(unsigned long long seed) {
    if (d_buffer_ != NULL)
        deallocate();
    d_buffer_ = (int*)devAlloc_->allocate(sizeof(int) * internalBufSize_);
    d_randStates_ = (curandStateMtgp32_t*)devAlloc_->allocate(sizeof(curandStateMtgp32_t) * CURAND_NUM_MTGP32_PARAMS);
    d_kernelParams_ = (mtgp32_kernel_params_t*)devAlloc_->allocate(sizeof(mtgp32_kernel_params_t) * CURAND_NUM_MTGP32_PARAMS);
    deviceRandomMakeKernelState(d_randStates_, d_kernelParams_, seed, stream_);
}

sqaod::SizeType DeviceRandom::getNRands() const {
    return (end_ - begin_ + internalBufSize_) % internalBufSize_;
}
    

void DeviceRandom::generate() {
    int nToGenerate = requiredSize_ - getNRands();
    nToGenerate = roundUp(nToGenerate, (int)randGenSize);
    if (0 <= nToGenerate) {
        deviceGenRand(d_buffer_, begin_, end_, nToGenerate, internalBufSize_, d_randStates_, stream_);
        end_ = (end_ + nToGenerate) % internalBufSize_;
    }
}

const int *DeviceRandom::get(sqaod::SizeType nRands, sqaod::IdxType *offset, sqaod::SizeType *posToWrap) {
    if (getNRands() < nRands)
        generate();
    assert(nRands <= getNRands());
      
    *offset = begin_;
    *posToWrap = internalBufSize_;
    begin_ = (begin_ + nRands) % internalBufSize_;
    return d_buffer_;
}

void DeviceRandom::synchronize() {
    throwOnError(cudaStreamSynchronize(stream_));
}