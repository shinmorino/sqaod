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
    bufSize_ = -1;
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
    bufSize_ = roundUp(nNums, mega) * 2;
    d_buffer_ = (int*)devAlloc_->allocate(sizeof(int) * bufSize_);
}
        

void DeviceRandom::deallocate() {
    cudaFree(d_buffer_);
    d_buffer_ = NULL;
}

void DeviceRandom::setSeed(unsigned long long seed) {
    if (d_randStates_ == NULL)
        d_randStates_ = (curandStateMtgp32_t*)devAlloc_->allocate(sizeof(curandStateMtgp32_t) * 200);
    if (d_kernelParams_ == NULL)
        d_kernelParams_ =
                (mtgp32_kernel_params_t*)devAlloc_->allocate(sizeof(mtgp32_kernel_params_t) * 200);
    deviceRandomMakeKernelState(d_randStates_, d_kernelParams_, seed, stream_);
}

sqaod::SizeType DeviceRandom::getNRands() const {
    return (end_ - begin_ + bufSize_) % bufSize_;
}
    

void DeviceRandom::generate() {
    /* mega must be a multiple of 51200 ( = 256 * 20 ) */
    int nToGenerate = bufSize_ - roundUp(getNRands(), (sqaod::SizeType)randsGenSize);
    end_ = (end_ + nToGenerate) % bufSize_;
}

const int *DeviceRandom::get(sqaod::SizeType nRands, sqaod::IdxType *offset) {
    if (getNRands() < nRands)
        generate();
    assert(getNRands() < nRands);
      
    *offset = begin_;
    begin_ = (begin_ + nRands) % bufSize_;
    return d_buffer_;
}
