#include "Common.h"
#include <iostream>
#include <float.h>

#ifndef SQAODC_CUDA_ENABLED
bool sqaod::isCUDAAvailable() {
    return false;
}

#else

#ifdef __linux__
#include <dlfcn.h>
bool sqaod::isCUDAAvailable() {
    void *h = dlopen("libcuda.so", RTLD_NOW);
    if (h == NULL)
        return false;
     /* shared library found */
    typedef int(*cuInitType)(unsigned int);
    typedef int(*cuDeviceGetCountType)(int *);
    cuInitType cuInit = (cuInitType)dlsym(h, "cuInit");
    cuDeviceGetCountType cuDeviceGetCount = (cuDeviceGetCountType)dlsym(h, "cuDeviceGetCount");

    bool deviceFound = false;
    int res = cuInit(0);
    if (res == 0) {
        int count = 0;
        res = cuDeviceGetCount(&count);
        deviceFound = (res == 0) && (count != 0);
    }
    dlclose(h);
    return deviceFound;
}

#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

bool sqaod::isCUDAAvailable() {
    HMODULE h = LoadLibrary(L"nvcuda.dll");
    if (h == NULL)
        return false;
    /* shared library found */
    typedef int(*cuInitType)(unsigned int);
    typedef int (*cuDeviceGetCountType)(int *);
    cuInitType cuInit = (cuInitType)GetProcAddress(h, "cuInit");
    cuDeviceGetCountType cuDeviceGetCount = (cuDeviceGetCountType)GetProcAddress(h, "cuDeviceGetCount");
    bool deviceFound = false;
    int res = cuInit(0);
    if (res == 0) {
        int count = 0;
        res = cuDeviceGetCount(&count);
        deviceFound = (res == 0) && (count != 0);
    }
    FreeLibrary(h);
    return deviceFound;
}
#endif

#endif

template<class V>
void sqaod::createBitSetSequence(V *bits, SizeType stride,
                                 SizeType nBits, PackedBitSet bBegin, PackedBitSet bEnd) {
    for (PackedBitSet b = bBegin; b < bEnd; ++b) {
        for (int pos = 0; pos < nBits; ++pos)
            bits[pos] = V((b >> (nBits - 1 - pos)) & 1);
        bits += stride;
    }
}

void sqaod::unpackBitSet(BitSet *unpacked, const PackedBitSet packed, int N) {
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
void ::sqaod::createBitSetSequence<double>(double *bits, sqaod::SizeType, sqaod::SizeType, PackedBitSet, PackedBitSet);
template
void ::sqaod::createBitSetSequence<float>(float *bits, sqaod::SizeType, sqaod::SizeType, PackedBitSet, PackedBitSet);
template
void ::sqaod::createBitSetSequence<char>(char *bits, sqaod::SizeType, sqaod::SizeType, PackedBitSet, PackedBitSet);

template
bool ::sqaod::isSymmetric<float>(const sqaod::MatrixType<float> &W);
template
bool ::sqaod::isSymmetric<double>(const sqaod::MatrixType<double> &W);

// template
// sqaod::MatrixType<double> sqaod::bitsToMat<double>(const BitMatrix &bits);
// template
// sqaod::MatrixType<float> sqaod::bitsToMat<float>(const BitMatrix &bits);
