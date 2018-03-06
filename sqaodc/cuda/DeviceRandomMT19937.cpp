#include "DeviceRandomMT19937.h"
#include "cudafuncs.h"
#include <common/Random.h>
#include <algorithm>
#include <time.h>

using namespace sqaod_cuda;
    
enum {
    randGenSize = 20 * (1 << 20)
};


DeviceRandomMT19937::DeviceRandomMT19937(Device &device, DeviceStream *devStream) {
    requiredSize_ = (sq::SizeType)-1;
    internalBufSize_ = (sq::SizeType)-1;
    gen_ = NULL;
    d_buffer_[0] = NULL;
    pos_ = 0;
    assignDevice(device, devStream);
}

DeviceRandomMT19937::DeviceRandomMT19937() {
    requiredSize_ = (sq::SizeType)-1;
    internalBufSize_ = (sq::SizeType) - 1;
    d_buffer_[0] = NULL;
    pos_ = 0;
}

DeviceRandomMT19937::~DeviceRandomMT19937() {
    if (d_buffer_[0] != NULL)
        deallocate();
    if (gen_ != NULL)
        throwOnError(curandDestroyGenerator(gen_));
    gen_ = NULL;
}

void DeviceRandomMT19937::assignDevice(Device &device, DeviceStream *devStream) {
    devAlloc_ = device.objectAllocator();
    if (devStream == NULL)
        devStream = device.defaultStream();
    stream_ = devStream->getCudaStream();
    throwOnError(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_MT19937));
    throwOnError(curandSetStream(gen_, stream_));
}


void DeviceRandomMT19937::setRequiredSize(sq::SizeType requiredSize) {
    /* Should give 2 chunks, 1 is for roundUp(), other is not to make size == 0 when filled up. */
    int newInternalBufSize = std::max(requiredSize, (sq::SizeType)randGenSize);
    if (newInternalBufSize != internalBufSize_) {
        internalBufSize_ = newInternalBufSize;
        if (d_buffer_[0] != NULL)
            devAlloc_->deallocate(d_buffer_[0]);
        devAlloc_->allocate(&d_buffer_[0], internalBufSize_ * 2);
        d_buffer_[1] = &d_buffer_[0][internalBufSize_];
        activePlane_ = 0;
        pos_ = internalBufSize_; /* set no random numbers in buffer */
    }
    requiredSize_ = requiredSize;
}


void DeviceRandomMT19937::deallocate() {
    devAlloc_->deallocate(d_buffer_[0]);
    d_buffer_[0] = NULL;
}

void DeviceRandomMT19937::seed(unsigned int seed) {
    curandSetPseudoRandomGeneratorSeed(gen_, seed);
}

void DeviceRandomMT19937::seed() {
    seed((unsigned long long)time(NULL));
}

sq::SizeType DeviceRandomMT19937::getNRands() const {
    return (sq::SizeType)(internalBufSize_ - pos_);
}

void DeviceRandomMT19937::generate() {
    throwErrorIf(internalBufSize_ == (sq::SizeType) -1, "DeviceRandom not initialized.");

    int nRands = getNRands();
    int nToGenerate = internalBufSize_ - nRands;
    int prevPlane = activePlane_;
    activePlane_ ^= 1;
#if 1
    throwOnError(cudaMemcpyAsync(d_buffer_[activePlane_], &d_buffer_[prevPlane][pos_],
                                    nRands * sizeof(unsigned int), cudaMemcpyDefault, stream_));
    throwOnError(curandGenerate(gen_, &d_buffer_[activePlane_][nRands], nToGenerate));
#else
    /* generate random numbers on CPU for validation. */
    synchronize();
    memmove(d_buffer_[activePlane_], &d_buffer_[prevPlane][pos_], nRands * sizeof(unsigned int));
    for (int idx = 0; idx < nToGenerate; ++idx) {
        d_buffer_[idx + nRands] = sq::random.randInt32();
#endif
    pos_ = 0;
}

const unsigned int *DeviceRandomMT19937::get(sq::SizeType nRands,
                                            sq::IdxType *offset, sq::SizeType *posToWrap, int alignment) {
    nRands = roundUp(nRands, (sq::SizeType)alignment);
    if (getNRands() < nRands)
        generate();
    assert(nRands <= getNRands());

    *offset = pos_;
    *posToWrap = internalBufSize_;
    pos_ += nRands;
    return d_buffer_[activePlane_];
}

void DeviceRandomMT19937::synchronize() {
    throwOnError(cudaStreamSynchronize(stream_));
}
