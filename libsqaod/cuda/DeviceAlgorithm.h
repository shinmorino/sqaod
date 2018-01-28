#ifndef CUDA_DEVICE_ALGORITHM_H__
#define CUDA_DEVICE_ALGORITHM_H__

#include <common/defines.h>
#include <cuda/DeviceStream.h>

namespace sqaod_cuda {

class Device;

struct DeviceAlgorithm {

// void sort(DeviceBitMatrix *bits);

// void sort(sqaod::CUDABitMatrix *bits0, sqaod::CUDABitMatrix *bits1);

// void sort(sqaod::CUDAPackedBitsArray *packedBitsArray);
    
    template<class V>
    void generateBitsSequence(V *d_data, int N,
                              sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);
    
    template<class real>
    void partition_Emin(sqaod::PackedBits *d_bitsListMin, sqaod::SizeType *d_nMin,
                        const real Emin,  const real *d_Ebatch,
                        const sqaod::PackedBits *d_bitsList, sqaod::SizeType len);

    template<class V>
    void randomize(V *d_data, const int *d_random, sqaod::SizeType N);
    
    
    void assignDevice(Device &device, DeviceStream *devStream = NULL);
    
private:
    cudaStream_t stream_;
    DeviceStream *devStream_;
};

}

#endif

