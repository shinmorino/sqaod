#ifndef CUDA_DEVICE_H__
#define CUDA_DEVICE_H__

#include <common/Array.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceMemoryStore.h>
#include <cuda/DeviceObjectAllocator.h>
#include <cuda/DeviceConstScalars.h>

namespace sqaod_cuda {

class DeviceStream;

class Device {
public:
    Device(int devNo = -1);
    ~Device();

    void initialize(int devNo = 0);
    void finalize();

    /* FIXME: add activate method. */

    DeviceObjectAllocator *objectAllocator();

    template<class real>
    const DeviceConstScalarsType<real> *constScalars() const;
    
    DeviceStream *newStream();

    DeviceStream *defaultStream();

    void releaseStream(DeviceStream *stream);

    /* sync on device */
    void synchronize();
    
private:
    int devNo_;
    DeviceMemoryStore memStore_;
    typedef sqaod::ArrayType<DeviceStream*> Streams;
    Streams streams_;

    /* Object allocators */
    DeviceObjectAllocator devObjAllocator_;
    DeviceStream defaultDeviceStream_;
    DeviceConstScalarsType<double> devConstScalarsFP64_;
    DeviceConstScalarsType<float> devConstScalarsFP32_;
};


inline
DeviceObjectAllocator *Device::objectAllocator() {
    return &devObjAllocator_;
}

template<> inline
const DeviceConstScalarsType<double> *Device::constScalars<double>() const {
    return &devConstScalarsFP64_;
}

template<> inline
const DeviceConstScalarsType<float> *Device::constScalars<float>() const {
    return &devConstScalarsFP32_;
}


}

#endif
