#ifndef CUDA_DEVICE_H__
#define CUDA_DEVICE_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceMemoryStore.h>
#include <vector>

namespace sqaod_cuda {

class DeviceStream;

class Device {
public:
    void setDevice(int devNo);
    
    template<class V>
    void allocate(DeviceMatrixType<V> *mat, int rows, int cols);

    template<class V>
    void allocate(DeviceMatrixType<V> *mat, const sqaod::Dim &dim);

    template<class V>
    void allocate(DeviceVectorType<V> *vec, int size);

    template<class V>
    void allocate(DeviceScalarType<V> *mat);

    void deallocate(DeviceObject *obj);

    /* Device Const */
    template<class V>
    const DeviceScalarType<V> &deviceConst(V c);
    template<class V>
    const DeviceScalarType<V> &d_one();
    template<class V>
    const DeviceScalarType<V> &d_zero();
    
    DeviceStream *newDeviceStream();

    DeviceStream *defaultDeviceStream();

    void releaseStream(DeviceStream *stream);

    /* sync on device */
    void synchronize();

    DeviceMemoryStore &memStore() {
        return memStore_;
    }
    
private:
    int devNo_;
    DeviceMemoryStore memStore_;
    typedef std::vector<DeviceStream *> Streams;
    Streams streams_;
};

}

#endif
