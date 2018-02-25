#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sqaodc/cuda/DeviceMemoryStore.h>


namespace sqaod_cuda {

namespace sq = sqaod;

template<class real> struct DeviceMatrixType;
template<class real> struct DeviceVectorType;
template<class real> struct DeviceScalarType;


class DeviceStream {
    friend class Device;

    DeviceStream();
    DeviceStream(cudaStream_t stream, DeviceMemoryStore &memStore, int nThreadsToFillDevice);
    ~DeviceStream();
    
    void set(cudaStream_t stream, DeviceMemoryStore &memStore, int nThreadsToFillDevice);

public:
    int getNumThreadsToFillDevice() const { return nThreadsToFillDevice_;  }

    void finalize();
    
    void *allocate(size_t size, const char *signature = NULL);

    template<class V>
    void allocate(V **pv, sq::SizeType size, const char *signature = NULL);

    template<class V>
    DeviceMatrixType<V> *tempDeviceMatrix(sq::SizeType rows, sq::SizeType cols, const char *signature = NULL);

    template<class V>
    DeviceMatrixType<V> *tempDeviceMatrix(const sq::Dim &dim, const char *signature = NULL);

    template<class V>
    DeviceVectorType<V> *tempDeviceVector(sq::SizeType size, const char *signature = NULL);

    template<class V>
    DeviceScalarType<V> *tempDeviceScalar(const char *signature = NULL);

    cudaStream_t getCudaStream() const {
        return stream_;
    }

    cublasHandle_t getCublasHandle() const {
        return cublasHandle_;
    }
    
    /* sync on stream */
    void synchronize();

private:    
    void releaseTempObjects();

    int nThreadsToFillDevice_;
    cudaStream_t stream_;
    DeviceMemoryStore *memStore_;
    typedef sq::ArrayType<DeviceObject*> DeviceObjects;
    DeviceObjects tempObjects_;
    cublasHandle_t cublasHandle_;
};


template<class V>
void DeviceStream::allocate(V **pv, sq::SizeType size, const char *signature) {
    *pv = (V*)allocate(sizeof(V) * size, signature);
}

template<class V> inline
DeviceMatrixType<V> *DeviceStream::tempDeviceMatrix(sq::SizeType rows, sq::SizeType cols, const char *signature) {
    void *d_pv = memStore_->allocate(sizeof(V) * rows * cols);
    DeviceMatrixType<V> *mat = new DeviceMatrixType<V>((V*)d_pv, rows, cols);
    tempObjects_.pushBack(mat);
    return mat;
}

template<class V> inline
DeviceMatrixType<V> *DeviceStream::tempDeviceMatrix(const sq::Dim &dim, const char *signature) {
    return tempDeviceMatrix<V>(dim.rows, dim.cols, signature);
}

template<class V> inline
DeviceVectorType<V> *DeviceStream::tempDeviceVector(sq::SizeType size, const char *signature) {
    void *d_pv = memStore_->allocate(sizeof(V) * size);
    DeviceVectorType<V> *vec = new DeviceVectorType<V>((V*)d_pv, size);
    tempObjects_.pushBack(vec);
    return vec;
}

template<class V> inline
DeviceScalarType<V> *DeviceStream::tempDeviceScalar(const char *signature) {
    void *d_pv = memStore_->allocate(sizeof(V));
    DeviceScalarType<V> *s = new DeviceScalarType<V>((V*)d_pv);
    tempObjects_.pushBack(s);
    return s;
}

}
