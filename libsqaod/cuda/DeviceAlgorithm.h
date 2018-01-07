#ifndef CUDA_DEVICE_ALGORITHM_H__
#define CUDA_DEVICE_ALGORITHM_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>


namespace sqaod_cuda {


// void sort(DeviceBitMatrix *bits);

// void sort(sqaod::CUDABitMatrix *bits0, sqaod::CUDABitMatrix *bits1);

// void sort(sqaod::CUDAPackedBitsArray *packedBitsArray);


void createBitsSequence(DeviceBitMatrix *bitsSequence, int N,
                        sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);


template<class real>
void bitsToMatrix(DeviceMatrixType<real> *mat, const DeviceBitMatrix &bits);

template<class real>
void bitsToVector(DeviceVectorType<real> *mat, const DeviceBitMatrix &bits);

template<class real>
void bitsFromMatrix(DeviceBits *bits, const DeviceMatrixType<real> &mat);

template<class real>
void bitsFromVector(DeviceBits *bits, const DeviceVectorType<real> &mat);
        
}


#endif

