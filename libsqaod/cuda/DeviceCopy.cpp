#include "DeviceCopy.h"
#include "cudafuncs.h"
#include "Device.h"

using namespace sqaod_cuda;


DeviceCopy::DeviceCopy() {
    devAlloc_ = NULL;
    stream_ = NULL;
}

DeviceCopy::DeviceCopy(Device &device, DeviceStream *devStream) {
    assignDevice(device, devStream);
}

void DeviceCopy::assignDevice(Device &device, DeviceStream *devStream) {
    devAlloc_ = device.objectAllocator();
    if (devStream == NULL)
        devStream = device.defaultStream();
    kernels_.assignStream(devStream);
    stream_ = devStream->getCudaStream();
}

void DeviceCopy::synchronize() const {
    throwOnError(cudaStreamSynchronize(stream_));
}
