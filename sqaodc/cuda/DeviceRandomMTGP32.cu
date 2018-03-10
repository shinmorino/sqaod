#include "DeviceRandomMTGP32.h"
#include "cudafuncs.h"
#include <curand_mtgp32_host.h>
#include <curand_mtgp32_kernel.h>
#include <common/Random.h>


/* FIXME: use multiple states utilize more threads.
 *         use memory store */

using namespace sqaod_cuda;
    
enum {
    randGenSize = CURAND_NUM_MTGP32_PARAMS * THREAD_NUM
};


DeviceRandomMTGP32::DeviceRandomMTGP32(Device &device, DeviceStream *devStream) {
    requiredSize_ = -1;
    internalBufSize_ = -1;
    d_buffer_ = NULL;
    d_randStates_ = NULL;
    d_kernelParams_ = NULL;
    begin_ = end_ = 0;
    assignDevice(device, devStream);
}

DeviceRandomMTGP32::DeviceRandomMTGP32() {
    requiredSize_ = -1;
    internalBufSize_ = -1;
    d_buffer_ = NULL;
    d_randStates_ = NULL;
    d_kernelParams_ = NULL;
    begin_ = end_ = 0;
}

DeviceRandomMTGP32::~DeviceRandomMTGP32() {
    if (d_buffer_ != NULL)
        deallocate();
}

void DeviceRandomMTGP32::assignDevice(Device &device, DeviceStream *devStream) {
    devAlloc_ = device.objectAllocator();
    if (devStream == NULL)
        devStream = device.defaultStream();
    stream_ = devStream->getCudaStream();
}


void DeviceRandomMTGP32::setRequiredSize(sq::SizeType requiredSize) {
    /* Should give 2 chunks, 1 is for roundUp(), other is not to make size == 0 when filled up. */
    int newInternalBufSize = roundUp(requiredSize, randGenSize)+ randGenSize * 2;
    if (newInternalBufSize != internalBufSize_) {
        internalBufSize_ = newInternalBufSize;
        if (d_buffer_ != NULL)
            devAlloc_->deallocate(d_buffer_);
        d_buffer_ = NULL;
    }
    requiredSize_ = requiredSize;
}
        

void DeviceRandomMTGP32::deallocate() {
    deallocateStates();
    deallocateBuffer();
}

void DeviceRandomMTGP32::deallocateStates() {
    devAlloc_->deallocate(d_randStates_);
    devAlloc_->deallocate(d_kernelParams_);
    d_randStates_ = NULL;
    d_kernelParams_ = NULL;
}

void DeviceRandomMTGP32::deallocateBuffer() {
    devAlloc_->deallocate(d_buffer_);
    d_buffer_ = NULL;
}

void DeviceRandomMTGP32::seed(unsigned int seed) {
    if (d_randStates_ != NULL)
        deallocateStates();
    devAlloc_->allocate(&d_randStates_, CURAND_NUM_MTGP32_PARAMS);
    devAlloc_->allocate(&d_kernelParams_, CURAND_NUM_MTGP32_PARAMS);
    /* synchronous */
    throwOnError(curandMakeMTGP32KernelState(
                         d_randStates_, MTGPDC_PARAM_TABLE,
                         d_kernelParams_, CURAND_NUM_MTGP32_PARAMS, seed));
}

void DeviceRandomMTGP32::seed() {
    seed((unsigned long)time(NULL));
}

sq::SizeType DeviceRandomMTGP32::getNRands() const {
    return (end_ - begin_ + internalBufSize_) % internalBufSize_;
}


__global__
static void genRandKernel(unsigned int *d_buffer, int offset, int nNums, int bufLen,
                          curandStateMtgp32_t *d_state) {
    /* bufLen must be 2^n */
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    offset = (offset + gid) % bufLen;
    for (int idx = 0; idx < nNums; idx += randGenSize) {
        unsigned int r = curand(&d_state[blockIdx.x]);
        d_buffer[offset] = r;
        offset = (offset + randGenSize) % bufLen;
    }
}


void DeviceRandomMTGP32::generate() {
    throwErrorIf(internalBufSize_ == -1, "DeviceRandom not initialized.");
    if (d_buffer_ == NULL)
        devAlloc_->allocate(&d_buffer_, internalBufSize_);

    int nToGenerate = requiredSize_ - getNRands();
    nToGenerate = roundUp(nToGenerate, (int)randGenSize);
    if (0 <= nToGenerate) {
#if 1
        genRandKernel<<<CURAND_NUM_MTGP32_PARAMS, THREAD_NUM, 0, stream_>>>
                (d_buffer_, end_, nToGenerate, internalBufSize_, d_randStates_);
        DEBUG_SYNC;
#else
        /* generate random numbers on CPU for validation. */
        synchronize();
        for (int idx = 0; idx < nToGenerate; ++idx)
            d_buffer_[(end_ + idx) % internalBufSize_] = sq::random.randInt32();
#endif
        end_ = (end_ + nToGenerate) % internalBufSize_;
    }
}

const unsigned int *DeviceRandomMTGP32::get(sq::SizeType nRands,
                                            sq::IdxType *offset, sq::SizeType *posToWrap, int alignment) {
    nRands = roundUp(nRands, alignment);
    if (getNRands() < nRands)
        generate();
    assert(nRands <= getNRands());

    *offset = begin_;
    *posToWrap = internalBufSize_;
    begin_ = (begin_ + nRands) % internalBufSize_;
    return d_buffer_;
}

void DeviceRandomMTGP32::synchronize() {
    throwOnError(cudaStreamSynchronize(stream_));
}
