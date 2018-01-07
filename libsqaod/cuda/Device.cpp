#include "Device.h"
#include "DeviceStream.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

void Device::setDevice(int devNo){
    devNo_ = devNo;
}

template<class V>
void Device::allocate(DeviceMatrixType<V> *mat, int rows, int cols) {
    mat->d_data = (V*)memStore_.allocate(sizeof(V) * rows * cols);
    mat->rows = rows;
    mat->cols = cols;
}

template<class V>
void Device::allocate(DeviceVectorType<V> *vec, int size) {
    vec->d_data = (V*)memStore_.allocate(sizeof(V) * size);
    vec->size = size;
}

template<class V>
void Device::allocate(DeviceScalarType<V> *sc) {
    sc->d_data = (V*)memStore_.allocate(sizeof(V));
}

void Device::deallocate(DeviceObject *obj) {
    void *pv = obj->get_data();
    memStore_.deallocate(pv);
}

DeviceStream *Device::newDeviceStream() {
    cudaStream_t stream;
    throwOnError(cudaStreamCreate(&stream));
    DeviceStream *deviceStream = new DeviceStream(stream, memStore_);
    streams_.push_back(deviceStream);
    return deviceStream;
}

DeviceStream *Device::defaultDeviceStream() {
    DeviceStream *stream = new DeviceStream(NULL, memStore_);
    streams_.push_back(stream);
    return stream;
}

void Device::releaseStream(DeviceStream *stream) {
    stream->releaseTempObjects();
    Streams::iterator it = std::find(streams_.begin(), streams_.end(), stream);
    assert(it != streams_.end());
    streams_.erase(it);
    delete stream;
}

/* sync on device */
void Device::synchronize() {
    throwOnError(cudaDeviceSynchronize());
    for (Streams::iterator it = streams_.begin(); it != streams_.end(); ++it) {
        (*it)->releaseTempObjects();
    }
}
