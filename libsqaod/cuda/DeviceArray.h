#ifndef SQAOD_DEVICEARRAY_H__
#define SQAOD_DEVICEARRAY_H__

#include <common/Array.h>
#include <cuda/DeviceObject.h>

namespace sqaod_cuda {

template<class V>
struct DeviceArrayType : DeviceObject {
    DeviceArrayType() : d_data(nullptr), size((sqaod::SizeType)-1) { }
    
    DeviceArrayType(const DeviceArrayType<V>&& arr) noexcept {
        DeviceArrayType newArr;
        newArr.d_data = arr.d_data;
        size = arr.size;
        bufferSize = arr.bufferSize;
        arr.d_data = NULL;
    }
    
    V *d_data;
    size_t bufferSize;
    size_t size;

private:
    DeviceArrayType(const DeviceArrayType<V>&);
    virtual void get_data(void **ppv) { 
        *ppv = d_data;
        d_data = NULL;
        size = (sqaod::SizeType)-1;
    }
};

typedef DeviceArrayType<sqaod::PackedBits> DevicePackedBitsArray;
typedef DeviceArrayType<char> DeviceBitArray;

}


#endif
