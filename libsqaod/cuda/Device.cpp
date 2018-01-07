#include "Device.h"
#include "DeviceStream.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

void Device::initialize(int devNo){
    devNo_ = devNo;
    memStore_.initialize();
    devObjAllocatorFP32_.initialize(memStore_, defaultDeviceStream_);
    devObjAllocatorFP64_.initialize(memStore_, defaultDeviceStream_);
}

DeviceStream &Device::newDeviceStream() {
    cudaStream_t stream;
    throwOnError(cudaStreamCreate(&stream));
    DeviceStream *deviceStream = new DeviceStream(stream, memStore_);
    streams_.pushBack(deviceStream);
    return *deviceStream;
}

DeviceStream &Device::defaultDeviceStream() {
    return defaultDeviceStream_;
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
