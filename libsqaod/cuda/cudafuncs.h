#ifndef CUDA_CUDAFUNCS_H__
#define CUDA_CUDAFUNCS_H__

#include <stdlib.h>
#include <cuda_runtime.h>


#ifdef _DEBUG
#define DEBUG_SYNC {throwOnError(cudaGetLastError()); throwOnError(cudaDeviceSynchronize()); }
#else
#define DEBUG_SYNC
#endif


namespace sqaod_cuda {

#define throwOnError(expr) expr

void throwError(const char *message);


template<class V>
inline V divru(const V &v, const V &base) {
    return (v + base - 1) / base;
}

template<class V>
inline V roundUp(const V &v, const V &base) {
    return ((v + base - 1) / base) * base;
}



}


#endif
