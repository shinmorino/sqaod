#include "Device.h"
#include "DeviceStream.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

Device::Device(int devNo) {
    devNo_ = -1;
    if (devNo != -1)
        initialize(devNo);
}

Device::~Device() {
    if (devNo_ != -1)
        finalize();
}

void Device::initialize(int devNo){
    devNo_ = devNo;
    throwOnError(cudaSetDevice(devNo));
    memStore_.initialize();
    defaultDeviceStream_.set(NULL, memStore_);
    devObjAllocatorFP32_.initialize(&memStore_, &defaultDeviceStream_);
    devObjAllocatorFP64_.initialize(&memStore_, &defaultDeviceStream_);
}

void Device::finalize() {
    synchronize();
    for (Streams::iterator it = streams_.begin(); it != streams_.end(); ++it)
        delete *it;
    defaultDeviceStream_.finalize();
    streams_.clear();
    devObjAllocatorFP32_.finalize();
    devObjAllocatorFP64_.finalize();
    memStore_.finalize();
    devNo_ = -1;
}


DeviceStream *Device::newStream() {
    cudaStream_t stream;
    throwOnError(cudaStreamCreate(&stream));
    DeviceStream *deviceStream = new DeviceStream(stream, memStore_);
    streams_.pushBack(deviceStream);
    return deviceStream;
}

DeviceStream *Device::defaultStream() {
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
    defaultDeviceStream_.releaseTempObjects();
}
