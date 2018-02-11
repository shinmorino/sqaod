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
        capacity = arr.capacity;
        arr.d_data = nullptr;
        arr.size = (sqaod::SizeType)-1;
        arr.capacity = (sqaod::SizeType)-1;
    }

    V &operator[](sqaod::IdxType idx) {
        return d_data[idx];
    }

    const V &operator[](sqaod::IdxType idx) const {
        return d_data[idx];
    }

    V *d_data;
    sqaod::SizeType capacity;
    sqaod::SizeType size;

private:
    DeviceArrayType(const DeviceArrayType<V>&);
    virtual void get_data(void **ppv) { 
        *ppv = d_data;
        d_data = nullptr;
        size = (sqaod::SizeType)-1;
    }
};

typedef DeviceArrayType<sqaod::PackedBits> DevicePackedBitsArray;
typedef DeviceArrayType<sqaod::PackedBitsPair> DevicePackedBitsPairArray;
typedef DeviceArrayType<char> DeviceBitArray;

}


#endif
