#ifndef CUDA_DEVICE_H__
#define CUDA_DEVICE_H__

#include <common/Array.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceMemoryStore.h>
#include <cuda/DeviceObjectAllocator.h>

namespace sqaod_cuda {

class DeviceStream;

class Device {
public:
    Device(int devNo = -1);
    ~Device();

    template<class real>
    using ObjectAllocator = DeviceObjectAllocatorType<real>;

    void initialize(int devNo = 0);
    void finalize();

    /* FIXME: add activate method. */

    template<class real>
    DeviceObjectAllocatorType<real> *objectAllocator();
    
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
    DeviceObjectAllocatorType<float> devObjAllocatorFP32_;
    DeviceObjectAllocatorType<double> devObjAllocatorFP64_;
    DeviceStream defaultDeviceStream_;
};


template<> inline
DeviceObjectAllocatorType<float> *Device::objectAllocator<float>() {
    return &devObjAllocatorFP32_;
}

template<> inline
DeviceObjectAllocatorType<double> *Device::objectAllocator<double>() {
    return &devObjAllocatorFP64_;
}


}

#endif
