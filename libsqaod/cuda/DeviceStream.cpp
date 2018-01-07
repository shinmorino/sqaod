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

void *DeviceStream::allocate(size_t size) {
    void *d_pv = memStore_.allocate(size);
    tempObjects_.pushBack(new PlainDeviceObject(d_pv));
    return d_pv;
}

template<class V>
void DeviceStream::allocate(DeviceMatrixType<V> **mat, int rows, int cols,
                            const char *signature) {
    void *d_pv = memStore_.allocate(sizeof(V) * rows * cols);
    *mat = new DeviceMatrixType<V>((V*)d_pv, rows, cols);
    tempObjects_.pushBack(*mat);
}

template<class V>
void DeviceStream::allocate(DeviceVectorType<V> **vec, int size, const char *signature) {
    void *d_pv = memStore_.allocate(sizeof(V) * size);
    *vec = new DeviceVectorType<V>((V*)d_pv, size);
    tempObjects_.pushBack(*vec);
}

template<class V>
void DeviceStream::allocate(DeviceScalarType<V> **s, const char *signature) {
    void *d_pv = memStore_.allocate(sizeof(V));
    *s = new DeviceScalarType<V>((V*)d_pv);
    tempObjects_.pushBack(*s);
}

void DeviceStream::newEvent() {
    /* FIXME: add implementation */
    assert(!"Not implemented.");
}
    
void DeviceStream::eventRecord() {
    assert(!"Not implemented.");
}
    
void DeviceStream::waitEvent(cudaEvent_t event) {
    assert(!"Not implemented.");
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
    }
}


