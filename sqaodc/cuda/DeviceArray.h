#pragma once

#include <sqaodc/common/Array.h>
#include <sqaodc/cuda/DeviceObject.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class V>
struct DeviceArrayType : DeviceObject {
    DeviceArrayType() : d_data(nullptr), size(-1) { }
    
    DeviceArrayType(const DeviceArrayType<V>&& arr) noexcept {
        DeviceArrayType newArr;
        newArr.d_data = arr.d_data;
        size = arr.size;
        capacity = arr.capacity;
        arr.d_data = nullptr;
        arr.size = 0;
        arr.capacity = -1;
    }

    V &operator[](sq::IdxType idx) {
        return d_data[idx];
    }

    const V &operator[](sq::IdxType idx) const {
        return d_data[idx];
    }

    V *d_data;
    sq::SizeType capacity;
    sq::SizeType size;

private:
    DeviceArrayType(const DeviceArrayType<V>&);
    virtual void get_data(void **ppv) { 
        *ppv = d_data;
        d_data = nullptr;
        size = -1;
    }
};

typedef DeviceArrayType<sq::PackedBits> DevicePackedBitsArray;
typedef DeviceArrayType<sq::PackedBitsPair> DevicePackedBitsPairArray;
typedef DeviceArrayType<char> DeviceBitArray;

}
