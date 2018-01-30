#include "cudafuncs.h"
#include <common/defines.h>
#include <stdio.h>

void sqaod_cuda::
_throwError(cudaError_t status, const char *file, unsigned long line, const char *expr) {
    char msg[512];
    const char *errName = cudaGetErrorName(status);
    snprintf(msg, sizeof(msg), "%s(%d), %s.", errName, (int)status, expr);
    sqaod::_throwError(file, line, msg);
}

void sqaod_cuda::
_throwError(cublasStatus_t status, const char *file, unsigned long line, const char *expr) {
    char msg[512];
    snprintf(msg, sizeof(msg), "cublasStatus = %d, %s.", (int)status, expr);
    sqaod::_throwError(file, line, msg);
}

void sqaod_cuda::
_throwError(curandStatus_t status, const char *file, unsigned long line, const char *expr) {
    char msg[512];
    snprintf(msg, sizeof(msg), "curandStatus = %d, %s.", (int)status, expr);
    sqaod::_throwError(file, line, msg);
}
