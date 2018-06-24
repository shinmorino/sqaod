#pragma once

#include <sqaodc/cuda/DeviceRandom.h>

namespace sqaod_cuda {

class DeviceRandomBuffer {
public:
    DeviceRandomBuffer();
    DeviceRandomBuffer(Device &device, DeviceStream *devStream = NULL);

    ~DeviceRandomBuffer();

    void deallocate();

    void assignDevice(Device &device, DeviceStream *devStream = NULL);

    bool available(sq::SizeType size) const {
        size_t nElmsAvailable = sizeInElm_ - posInElm_;
        return size_t(size) <= nElmsAvailable;
    }

    void generateFlipPositions(DeviceRandom &d_random,
                               sq::SizeType N, sq::SizeType m, int nPlanes);

    template<class real>
    void generate(DeviceRandom &d_random, sq::SizeType size);

    void generateFloat(DeviceRandom &d_random, sq::SizeType size);
    void generateDouble(DeviceRandom &d_random, sq::SizeType size);

    template<class V>
    const V *acquire(sq::SizeType size) {
        const V *ptr = &((const V*)d_buffer_)[posInElm_];
        posInElm_ += size;
        return ptr;
    }

private:
    void reserve(size_t bufSize);

    sq::IdxType posInElm_;
    size_t sizeInElm_;
    size_t sizeInByte_;
    void *d_buffer_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};


template<> inline
void DeviceRandomBuffer::generate<float>(DeviceRandom &d_random, sq::SizeType size) {
    generateFloat(d_random, size);
}

template<> inline
void DeviceRandomBuffer::generate<double>(DeviceRandom &d_random, sq::SizeType size) {
    generateDouble(d_random, size);
}


}
