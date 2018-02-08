#include "DeviceRandomBuffer.h"
#include "DeviceRandom.cuh"
#include "cudafuncs.h"


using namespace sqaod_cuda;
namespace sq = sqaod;

DeviceRandomBuffer::DeviceRandomBuffer() {
    sizeInByte_ = (sq::SizeType)-1;
    sizeInElm_ = 0;
    posInElm_ = 0;
    d_buffer_ = NULL;
}

DeviceRandomBuffer::DeviceRandomBuffer(Device &device, DeviceStream *devStream) {
    sizeInByte_ = (sq::SizeType)-1;
    d_buffer_ = NULL;
    sizeInElm_ = 0;
    posInElm_ = 0;
    assignDevice(device, devStream);
}

DeviceRandomBuffer::~DeviceRandomBuffer() {
    deallocate();
}

void DeviceRandomBuffer::deallocate() {
    if (d_buffer_ != NULL) {
        devAlloc_->deallocate(d_buffer_);
        d_buffer_ = NULL;
        sizeInByte_ = (sq::SizeType)-1;
        sizeInElm_ = (sq::SizeType)-1;
    }
}

void DeviceRandomBuffer::assignDevice(Device &device, DeviceStream *devStream) {
    devAlloc_ = device.objectAllocator();
    if (devStream == NULL)
        devStream = device.defaultStream();
    devStream_ = devStream;
}

void DeviceRandomBuffer::reserve(sq::SizeType size) {
    if ((sizeInByte_ != size) && (d_buffer_ != NULL))
        deallocate();
    if (d_buffer_ == NULL) {
        d_buffer_ = devAlloc_->allocate(size);
        sizeInByte_ = size;
    }
}


/* generate random */
__global__ static void
generateFlipPosKernel(sq::IdxType *d_buffer, sq::SizeType N, sq::SizeType m, sq::SizeType nAllSteps,
                      const unsigned int *d_random, sq::IdxType offset, sq::SizeType posToWrap) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int iStep = gid / m;
    int iTrotter = gid % m;
    if (iStep < nAllSteps) {
        int posOffset = (iStep + iTrotter) & 1;
        d_buffer[gid] = (2 * d_random[(gid + offset) % posToWrap] + posOffset) % N;
    }
}

void DeviceRandomBuffer::generateFlipPositions(DeviceRandom &d_random,
                                               sqaod::SizeType N, sqaod::SizeType m,
                                               int nRuns) {
    int nToGenerate = N * m * nRuns;
    sq::SizeType size = nToGenerate * sizeof(int);
    reserve(size);
    sq::IdxType offset;
    sq::SizeType posToWrap;
    const unsigned int *d_randomNum = d_random.get(nToGenerate, &offset, &posToWrap);

    dim3 blockDim(128);
    dim3 gridDim(divru((sq::SizeType)nToGenerate, blockDim.x));
    cudaStream_t stream = devStream_->getCudaStream();
    generateFlipPosKernel<<<gridDim, blockDim, 0, stream>>>((int*)d_buffer_, N, m, N * nRuns,
                                                            d_randomNum, offset, posToWrap);
    DEBUG_SYNC;

    posInElm_ = 0;
    sizeInElm_ = nToGenerate;
}


__global__
static void genRandKernel(float *d_buffer, int nToGenerate,
                          const unsigned int *d_random, sq::IdxType offset, sq::SizeType posToWrap) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < nToGenerate) {
        unsigned int randnum = d_random[(gid + offset) % posToWrap];
        d_buffer[gid] = random(randnum);
    }
}

__global__
static void genRandKernel(double *d_buffer, int nToGenerate,
                          const uint2 *d_random, sq::IdxType offset, sq::SizeType posToWrap) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < nToGenerate) {
        uint2 randnum = d_random[(gid + offset) % posToWrap];
        d_buffer[gid] = random(randnum);
    }
}


void DeviceRandomBuffer::generateFloat(DeviceRandom &d_random, sqaod::SizeType nToGenerate) {
    reserve(nToGenerate * sizeof(float));
    dim3 blockDim(128);
    dim3 gridDim(divru(nToGenerate, blockDim.x));
    cudaStream_t stream = devStream_->getCudaStream();
    sq::IdxType offset;
    sq::SizeType posToWrap;
    const unsigned int *d_randNum = d_random.get(nToGenerate, &offset, &posToWrap);
    genRandKernel<<<gridDim, blockDim, 0, stream>>>((float*)d_buffer_, nToGenerate,
                                                    d_randNum, offset, posToWrap);
    DEBUG_SYNC;
    posInElm_ = 0;
    sizeInElm_ = nToGenerate;
}

void DeviceRandomBuffer::generateDouble(DeviceRandom &d_random, sqaod::SizeType nToGenerate) {
    reserve(nToGenerate * sizeof(double));
    dim3 blockDim(128);
    dim3 gridDim(divru(nToGenerate, blockDim.x));
    cudaStream_t stream = devStream_->getCudaStream();
    sq::IdxType offset;
    sq::SizeType posToWrap;
    /* 2 random numbers are used to generate one double random number. */
    const unsigned int *d_randNum = d_random.get(nToGenerate * 2, &offset, &posToWrap, 2);
    genRandKernel<<<gridDim, blockDim, 0, stream>>>((double*)d_buffer_, nToGenerate,
                                                    (const uint2*)d_randNum, offset, posToWrap);
    DEBUG_SYNC;
    posInElm_ = 0;
    sizeInElm_ = nToGenerate;
}
