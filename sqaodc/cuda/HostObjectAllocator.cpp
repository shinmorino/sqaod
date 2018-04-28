#include "HostObjectAllocator.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

void *HostObjectAllocator::allocate(size_t size) {
    void *pv;
    throwOnError(cudaHostAlloc(&pv, size, cudaHostAllocDefault));
    return pv;
}


void *HostObjectAllocator::allocate2d(sq::SizeType *stride, size_t width, size_t height) {
    /* Host memory alignment */
    const int alignment = 64;
    *stride = sq::roundUp(width, alignment);
    size_t size = *stride * height;
    
    void *pv;
    throwOnError(cudaHostAlloc(&pv, size, cudaHostAllocDefault));
    return pv;
}


void HostObjectAllocator::deallocate(void *pv) {
    throwOnError(cudaFreeHost(pv));
}
