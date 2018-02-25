#include "HostObjectAllocator.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

void *HostObjectAllocator::allocate(size_t size) {
    void *pv;
    throwOnError(cudaHostAlloc(&pv, size, cudaHostAllocDefault));
    return pv;
}

void HostObjectAllocator::deallocate(void *pv) {
    throwOnError(cudaFreeHost(pv));
}
