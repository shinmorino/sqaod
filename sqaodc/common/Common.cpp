#include "Common.h"
#include <iostream>
#include <float.h>

#ifdef __linux__
#include <dlfcn.h>

bool sqaod::isCUDAAvailable() {
#ifndef SQAODC_CUDA_ENABLED
    return false;
#else
    void *h = dlopen("libcuda.so", RTLD_NOW);
    if (h == NULL)
        return false;
    /* shared library found */
    typedef int (*cuDeviceGetCountType)(int *);
    cuDeviceGetCountType cuDeviceGetCount = (cuDeviceGetCountType)dlsym(h, "cuDeviceGetCount");
    int count = 0;
    int res = cuDeviceGetCount(&count);
    bool deviceFound = (res == 0) && (count != 0);
    dlclose(h);
    return deviceFound;
#endif
    return false;
}

#endif


template<class real>
void sqaod::createBitsSequence(real *bits, int nBits, PackedBits bBegin, PackedBits bEnd) {
    for (PackedBits b = bBegin; b < bEnd; ++b) {
        for (int pos = 0; pos < nBits; ++pos)
            bits[pos] = real((b >> (nBits - 1 - pos)) & 1);
        bits += nBits;
    }
}

void sqaod::unpackBits(Bits *unpacked, const PackedBits packed, int N) {
    unpacked->resize(N);
    for (int pos = 0; pos < N; ++pos) {
        (*unpacked)(pos) = (packed >> (N - 1 - pos)) & 1;
    }
}


template<class real>
bool sqaod::isSymmetric(const MatrixType<real> &W) {
    for (SizeType j = 0; j < W.rows; ++j) {
        for (SizeType i = 0; i < j + 1; ++i)
            if (W(i, j) != W(j, i))
                return false;
    }
    return true;
}


// template<class real>
// sqaod::MatrixType<real> sqaod::bitsToMat(const BitMatrix &bits) {
//     MatrixType<real> mat(bits.dim());
//     mat.map() = bits.map().cast<real>();
//     return mat;
// }


template
void ::sqaod::createBitsSequence<double>(double *bits, int nBits, PackedBits bBegin, PackedBits bEnd);
template
void ::sqaod::createBitsSequence<float>(float *bits, int nBits, PackedBits bBegin, PackedBits bEnd);
template
void ::sqaod::createBitsSequence<char>(char *bits, int nBits, PackedBits bBegin, PackedBits bEnd);

template
bool ::sqaod::isSymmetric<float>(const sqaod::MatrixType<float> &W);
template
bool ::sqaod::isSymmetric<double>(const sqaod::MatrixType<double> &W);

// template
// sqaod::MatrixType<double> sqaod::bitsToMat<double>(const BitMatrix &bits);
// template
// sqaod::MatrixType<float> sqaod::bitsToMat<float>(const BitMatrix &bits);
