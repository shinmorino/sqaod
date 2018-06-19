#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "os_dependent.h"

int sqaod::getDefaultNumThreads() {
#ifdef _OPENMP
    HANDLE h = GetCurrentProcess();
    DWORD_PTR processAffinityMask, systemAffinityMask;
    GetProcessAffinityMask(h, &processAffinityMask, &systemAffinityMask);
    return (int)__popcnt64(processAffinityMask);
#else
    return 1;
#endif
}

#include <malloc.h>

void *sqaod::aligned_alloc(int alignment, size_t size) {
    return ::_aligned_malloc(size, alignment);
}

void sqaod::aligned_free(void *pv) {
    ::_aligned_free(pv);
}
