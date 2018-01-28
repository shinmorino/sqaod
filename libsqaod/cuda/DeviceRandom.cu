#include "DeviceRandom.h"
#include "cudafuncs.h"
#include <curand_mtgp32_host.h>
#include <curand_mtgp32_kernel.h>


/* FIXME: use multiple states utilize more threads.
 *         use memory store */

using namespace sqaod_cuda;

static const sqaod::SizeType mega = 1024 * 1024;
    
DeviceRandom::DeviceRandom(Device &device, DeviceStream *devStream) {
    assignDevice(device, devStream);
}

DeviceRandom::DeviceRandom() {
    bufLen_ = -1;
    d_buffer_ = NULL;
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


void DeviceRandom::allocate(sqaod::SizeType nNums) {
    assert(d_buffer_ == NULL);
    bufLen_ = roundUp(nNums, mega) * 2;
    d_buffer_ = (int*)devAlloc_->allocate(sizeof(int) * bufLen_);
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

sqaod::SizeType DeviceRandom::getNRands() const {
    return (end_ - begin_ + bufLen_) % bufLen_;
}
    

void DeviceRandom::generate() {
    /* mega must be a multiple of 51200 ( = 256 * 20 ) */
    int nToGenerate = bufLen - roundUp(getNRands(), (sqaod::SizeType)randsGenSize);
    randGenKernel<<<20, 256, 0, stream_>>>(d_buffer_, end_, nToGenerate, bufLen_, d_randStates_);
    DEBUG_SYNC;
    end_ = (end_ + nToGenerate) % bufLen_;
}

const int *DeviceRandom::get(sqaod::SizeType nRands, sqaod::IdxType *offset) {
    if (getNRands() < nRands)
        generate();
    assert(getNRands() < nRands);
      
    *offset = begin_;
    begin_ = (begin_ + nRands) % bufLen_;
    return d_buffer_;
}
