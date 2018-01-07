#ifndef SQAOD_DEVICEARRAY_H__
#define SQAOD_DEVICEARRAY_H__

#include <common/Array.h>

namespace sqaod_cuda {

template<class V>
struct DeviceArrayType {
    typedef DeviceArrayType<V> DeviceArray;
    DeviceArrayType();

    void append(const DeviceArray &dar);
    
    static
    void swap(DeviceArray *lhs, DeviceArray *rhs);

    // friend
    // void merge(DevicePackedBitsArray *lhs, DevicePackedBitsArray *rhs);


    V *d_data;
    size_t size;
};

typedef DeviceArrayType<sqaod::PackedBits> DevicePackedBitsArray;
typedef DeviceArrayType<char> DeviceBitArray;

}


#endif
