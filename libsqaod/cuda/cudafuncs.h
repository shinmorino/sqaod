#ifndef CUDA_CUDAFUNCS_H__
#define CUDA_CUDAFUNCS_H__

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iterator>

/* FIXME: undef somewhere. */
#ifdef _DEBUG
#define DEBUG_SYNC {throwOnError(cudaGetLastError()); throwOnError(cudaDeviceSynchronize()); }
#else
#define DEBUG_SYNC
#endif

#ifdef _DEBUG
#define CUB_DEBUG (false)
#else
#define CUB_DEBUG (false)
#endif


namespace sqaod_cuda {

inline bool _valid(cudaError_t cuerr) { return cuerr == cudaSuccess; }
void _throwError(cudaError_t status, const char *file, unsigned long line, const char *expr);

inline bool _valid(cublasStatus_t cublasStatus) { return cublasStatus == CUBLAS_STATUS_SUCCESS; }
void _throwError(cublasStatus_t status, const char *file, unsigned long line, const char *expr);

inline bool _valid(curandStatus_t cublasStatus) { return cublasStatus == CURAND_STATUS_SUCCESS; }
void _throwError(curandStatus_t status, const char *file, unsigned long line, const char *expr);

#define throwOnError(expr) { auto status = (expr); if (!sqaod_cuda::_valid(status)) { sqaod_cuda::_throwError(status, __FILE__, __LINE__, #expr); } }



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
