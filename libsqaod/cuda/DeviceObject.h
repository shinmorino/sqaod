#ifndef SQAOD_CUDA_DEVICEOBJECT_H__
#define SQAOD_CUDA_DEVICEOBJECT_H__

namespace sqaod_cuda {

struct DeviceObject {
    virtual ~DeviceObject() { }
    virtual void get_data(void **ppv) = 0;
};

}

#endif
