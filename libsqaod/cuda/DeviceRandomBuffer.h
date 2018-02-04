#pragma once

#include <cuda/DeviceRandom.h>

namespace sqaod_cuda {

class DeviceRandomBuffer {
public:
    DeviceRandomBuffer();
    DeviceRandomBuffer(Device &device, DeviceStream *devStream = NULL);

    ~DeviceRandomBuffer();

    void assignDevice(Device &device, DeviceStream *devStream = NULL);

    bool available(sqaod::SizeType size) const {
        int nElmsAvailable = sizeInElm_ - posInElm_;
        return (int)size <= nElmsAvailable;
    }

    void generateFlipPositions(DeviceRandom &d_random,
                               sqaod::SizeType N, sqaod::SizeType m, int nPlanes);

    template<class real>
    void generate(DeviceRandom &d_random, sqaod::SizeType size);

    void generateFloat(DeviceRandom &d_random, sqaod::SizeType size);
    void generateDouble(DeviceRandom &d_random, sqaod::SizeType size);

    template<class V>
    const V *acquire(sqaod::SizeType size) {
        const V *ptr = &((const V*)d_buffer_)[posInElm_];
        posInElm_ += size;
        return ptr;
    }

private:
    void deallocate();
    void reserve(sqaod::SizeType bufSize);

    sqaod::IdxType posInElm_;
    sqaod::SizeType sizeInElm_;
    sqaod::SizeType sizeInByte_;
    void *d_buffer_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};


template<> inline
void DeviceRandomBuffer::generate<float>(DeviceRandom &d_random, sqaod::SizeType size) {
    generateFloat(d_random, size);
}

template<> inline
void DeviceRandomBuffer::generate<double>(DeviceRandom &d_random, sqaod::SizeType size) {
    generateDouble(d_random, size);
}


}
