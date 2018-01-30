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

    template<class V>
    void createXListAndFlags(V *d_data, int N,
                             sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);

    template<class real>
    void partition(sqaod::PackedBits *d_out, sqaod::SizeType *d_nOut,
                   const sqaod::PackedBits *d_in, const char *d_flags, sqaod::SizeType nIn);

    template<class V>
    void randomize(V *d_data, const int *d_random, sqaod::SizeType N);
    

    DeviceAlgorithm(DeviceStream *devStream = NULL);

    void assignStream(DeviceStream *devStream);
    
private:
    cudaStream_t stream_;
    DeviceStream *devStream_;
};

}

#endif

