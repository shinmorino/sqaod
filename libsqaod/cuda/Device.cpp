#include "Device.h"
#include "DeviceStream.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

void Device::setDevice(int devNo){
    devNo_ = devNo;
}

DeviceStream *Device::newDeviceStream() {
    cudaStream_t stream;
    throwOnError(cudaStreamCreate(&stream));
    DeviceStream *deviceStream = new DeviceStream(stream, memStore_);
    streams_.pushBack(deviceStream);
    return deviceStream;
}

DeviceStream *Device::defaultDeviceStream() {
    DeviceStream *stream = new DeviceStream(NULL, memStore_);
    streams_.pushBack(stream);
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
