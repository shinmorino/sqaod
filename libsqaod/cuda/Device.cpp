#include "Device.h"
#include "DeviceStream.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

void Device::initialize(int devNo){
    devNo_ = devNo;
    memStore_.initialize();
    defaultDeviceStream_.set(NULL, memStore_);
    devObjAllocatorFP32_.initialize(&memStore_, &defaultDeviceStream_);
    devObjAllocatorFP64_.initialize(&memStore_, &defaultDeviceStream_);
}

void Device::finalize() {
    for (Streams::iterator it = streams_.begin(); it != streams_.end(); ++it) {
        (*it)->finalize();
        delete *it;
    }
    streams_.clear();
    devObjAllocatorFP32_.finalize();
    devObjAllocatorFP64_.finalize();
    memStore_.finalize();
}


DeviceStream *Device::newDeviceStream() {
    cudaStream_t stream;
    throwOnError(cudaStreamCreate(&stream));
    DeviceStream *deviceStream = new DeviceStream(stream, memStore_);
    streams_.pushBack(deviceStream);
    return deviceStream;
}

DeviceStream *Device::defaultDeviceStream() {
    return &defaultDeviceStream_;
}

void Device::releaseStream(DeviceStream *stream) {
    stream->releaseTempObjects();
    Streams::iterator it = std::find(streams_.begin(), streams_.end(), stream);
    assert(it != streams_.end());
    streams_.erase(it);
    if (stream != &defaultDeviceStream_)
        delete stream;
}

/* sync on device */
void Device::synchronize() {
    throwOnError(cudaDeviceSynchronize());
    for (Streams::iterator it = streams_.begin(); it != streams_.end(); ++it) {
        (*it)->releaseTempObjects();
    }
}
