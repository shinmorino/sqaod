#include "DeviceStream.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

namespace {

struct PlainDeviceObject : DeviceObject {
    PlainDeviceObject(void *d_pv) : d_data(d_pv) { }
    virtual void *get_data() {
        return d_data;
    }
    void *d_data;
};

}


DeviceStream::DeviceStream(cudaStream_t stream, DeviceMemoryStore &memStore) :
        stream_(stream), memStore_(memStore) {
    throwOnError(cublasCreate(&cublasHandle_));
    throwOnError(cublasSetStream(cublasHandle_, stream));
}

DeviceStream::~DeviceStream() {
    throwOnError(cublasDestroy(cublasHandle_));
}

/* sync on stream */
void DeviceStream::synchronize() {
    throwOnError(cudaStreamSynchronize(stream_));
}

void DeviceStream::releaseTempObjects() {
    for (DeviceObjects::iterator it = tempObjects_.begin();
         it != tempObjects_.end(); ++it) {
        void *pv = (*it)->get_data();
        memStore_.deallocate(pv);
        delete *it;
    }
    tempObjects_.clear();
}

void *DeviceStream::allocate(size_t size) {
    void *d_pv = memStore_.allocate(size);
    tempObjects_.pushBack(new PlainDeviceObject(d_pv));
    return d_pv;
}

