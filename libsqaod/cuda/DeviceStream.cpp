#include <common/Matrix.h>
#include "DeviceStream.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

namespace {

struct PlainDeviceObject : DeviceObject {
    PlainDeviceObject(void *d_pv) : d_data(d_pv) { }
    void get_data(void **pv) {
        *pv = d_data;
    }
    void *d_data;
};

}

DeviceStream::DeviceStream() : stream_(NULL), memStore_(NULL), cublasHandle_(NULL) {
}

DeviceStream::DeviceStream(cudaStream_t stream, DeviceMemoryStore &memStore) :
        stream_(NULL), memStore_(NULL), cublasHandle_(NULL) {
    set(stream, memStore);
}

DeviceStream::~DeviceStream() {
    finalize();
}

void DeviceStream::set(cudaStream_t stream, DeviceMemoryStore &memStore) {
    stream_ = stream;
    memStore_ = &memStore;
    throwOnError(cublasCreate(&cublasHandle_));
    throwOnError(cublasSetStream(cublasHandle_, stream));
    cublasSetPointerMode(cublasHandle_, CUBLAS_POINTER_MODE_DEVICE);
}


void DeviceStream::finalize() {
    synchronize();
    if (cublasHandle_ != NULL)
        throwOnError(cublasDestroy(cublasHandle_));
    if (stream_ != NULL)
        throwOnError(cudaStreamDestroy(stream_));
    cublasHandle_ = NULL;
    stream_ = NULL;
}


/* sync on stream */
void DeviceStream::synchronize() {
    throwOnError(cudaStreamSynchronize(stream_));
    releaseTempObjects();
}

void DeviceStream::releaseTempObjects() {
    for (DeviceObjects::iterator it = tempObjects_.begin();
         it != tempObjects_.end(); ++it) {
        void *pv;
        (*it)->get_data(&pv);
        memStore_->deallocate(pv);
        delete *it;
    }
    tempObjects_.clear();
}

void *DeviceStream::allocate(size_t size, const char *signature) {
    void *d_pv = memStore_->allocate(size);
    tempObjects_.pushBack(new PlainDeviceObject(d_pv));
    return d_pv;
}

