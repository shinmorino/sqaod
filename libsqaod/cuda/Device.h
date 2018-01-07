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
    void initialize(int devNo);
    void uninitialize();

    template<class real>
    DeviceObjectAllocatorType<real> &deviceObjectAllocator();
    
    DeviceStream &newDeviceStream();

    DeviceStream &defaultDeviceStream();

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
DeviceObjectAllocatorType<float> &Device::deviceObjectAllocator<float>() {
    return devObjAllocatorFP32_;
}

template<> inline
DeviceObjectAllocatorType<double> &Device::deviceObjectAllocator<double>() {
    return devObjAllocatorFP64_;
}


}

#endif
