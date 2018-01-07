#ifndef CUDA_DEVICE_STREAM_H__
#define CUDA_DEVICE_STREAM_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/Device.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace sqaod_cuda {


class DeviceStream {
    friend class Device;
    DeviceStream(cudaStream_t stream, DeviceMemoryStore &memStore);
    ~DeviceStream();

public:

    void *allocate(size_t size);

    template<class V>
    void allocate(DeviceMatrixType<V> **mat, int rows, int cols, const char *signature = NULL);

    template<class V>
    void allocate(DeviceVectorType<V> **vec, int size, const char *signature = NULL);

    template<class V>
    void allocate(DeviceScalarType<V> **s, const char *signature = NULL);

    cudaStream_t getStream() const {
        return stream_;
    }

    cublasHandle_t getCublasHandle() const {
        return cublasHandle_;
    }
    
    /* sync on stream */
    void synchronize();

private:    
    void releaseTempObjects();

    cudaStream_t stream_;
    DeviceMemoryStore &memStore_;
    typedef sqaod::ArrayType<DeviceObject*> DeviceObjects;
    DeviceObjects tempObjects_;
    cublasHandle_t cublasHandle_;
};

}


#endif
