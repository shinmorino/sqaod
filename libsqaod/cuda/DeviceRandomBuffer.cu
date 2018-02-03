#include "DeviceRandomBuffer.h"
#include "DeviceRandom.cuh"
#include "cudafuncs.h"


using namespace sqaod_cuda;
namespace sq = sqaod;

DeviceRandomBuffer::DeviceRandomBuffer() {
    size_ = (sq::SizeType)-1;
    d_buffer_ = NULL;
}


DeviceRandomBuffer::~DeviceRandomBuffer() {
    deallocate();
}


void DeviceRandomBuffer::deallocate() {
    if (d_buffer_ != NULL) {
        devAlloc_->deallocate(d_buffer_);
        d_buffer_ = NULL;
        size_ = (sq::SizeType)-1;
    }
}


void DeviceRandomBuffer::reserve(sq::SizeType size) {
    if ((size_ != size) && (d_buffer_ != NULL))
        deallocate();
    if (d_buffer_ == NULL) {
        d_buffer_ = devAlloc_->allocate(size);
        size_ = size;
    }
}


/* generate random */
__global__ static void
generateFlipPosKernel(sq::IdxType *d_buffer, sq::SizeType N, sq::SizeType m, sq::SizeType nSteps,
                      const int *d_random, sq::IdxType offset, sq::SizeType posToWrap) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int iStep = gid / m;
    int iTrotter = gid % m;
    if (iStep < nSteps) {
        int stepOffset = iStep & 1;
        int trotterOffset = iTrotter & 1;
        int posOffset = (stepOffset + trotterOffset) & 1;
        d_buffer[gid] = (2 * d_random[(gid + offset) % posToWrap] + posOffset) % N;
    }
}

void DeviceRandomBuffer::generateFlipPositions(DeviceRandom &d_random,
                                               sqaod::SizeType N, sqaod::SizeType m,
                                               int nRuns) {
    int nToGenerate = N * m * nRuns;
    sq::SizeType size = nToGenerate * sizeof(int);
    reserve(size);

    dim3 blockDim(128);
    dim3 gridDim(divru((uint)nToGenerate, blockDim.x));
    sq::IdxType offset;
    sq::SizeType posToWrap;
    const int *d_randomNum = d_random.get(nToGenerate, &offset, &posToWrap);
    cudaStream_t stream = devStream_->getCudaStream();
    generateFlipPosKernel<<<gridDim, blockDim, 0, stream>>>((int*)d_buffer_, N, m, nRuns,
                                                            d_randomNum, offset, posToWrap);
    DEBUG_SYNC;
    pos_ = 0;
    elmSize_ = sizeof(int);
}


__global__
static void genRandKernel(float *d_buffer, int nToGenerate,
                          const int *d_random, sq::IdxType offset, sq::SizeType posToWrap) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (nToGenerate < gid) {
        int randnum = d_random[(gid + offset) % posToWrap];
        d_buffer[gid] = random(randnum);
    }
}

__global__
static void genRandKernel(double *d_buffer, int nToGenerate,
                          const int2 *d_random, sq::IdxType offset, sq::SizeType posToWrap) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (nToGenerate < gid) {
        int2 randnum = d_random[(gid + offset) % posToWrap];
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
    const int *d_randNum = d_random.get(nToGenerate, &offset, &posToWrap);
    genRandKernel<<<gridDim, blockDim, 0, stream>>>((float*)d_buffer_, nToGenerate,
                                                    d_randNum, offset, posToWrap);
    DEBUG_SYNC;
    pos_ = 0;
    elmSize_ = sizeof(float);
}

void DeviceRandomBuffer::generateDouble(DeviceRandom &d_random, sqaod::SizeType nToGenerate) {
    reserve(nToGenerate * sizeof(double));
    dim3 blockDim(128);
    dim3 gridDim(divru(nToGenerate, blockDim.x));
    cudaStream_t stream = devStream_->getCudaStream();
    sq::IdxType offset;
    sq::SizeType posToWrap;
    /* 2 random numbers are used to generate one double random number. */
    const int *d_randNum = d_random.get(nToGenerate * 2, &offset, &posToWrap, 2);
    genRandKernel<<<gridDim, blockDim, 0, stream>>>((double*)d_buffer_, nToGenerate,
                                                    (const int2*)d_randNum, offset, posToWrap);
    DEBUG_SYNC;
    pos_ = 0;
    elmSize_ = sizeof(double);
}
