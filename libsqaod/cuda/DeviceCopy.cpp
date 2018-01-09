#include "DeviceCopy.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;

template<class V>
void DeviceCopy::copy(V *d_buf, const V *v, size_t nElms) const {
    cudaStream_t stream = devStream_->getStream();
    throwOnError(cudaMemcpyAsync(d_buf, v, sizeof(V) * nElms, cudaMemcpyDefault, stream));
}

DeviceCopy::DeviceCopy() : devStream_(NULL) { }

DeviceCopy::DeviceCopy(DeviceStream &stream) : devStream_(&stream) { }
void DeviceCopy::setStream(DeviceStream &stream) {
    devStream_ = &stream;
}


template void DeviceCopy::copy(double *d_buf, const double *v, size_t nElms) const;
template void DeviceCopy::copy(float *d_buf, const float *v, size_t nElms) const;
template void DeviceCopy::copy(char *d_buf, const char *v, size_t nElms) const;
template void DeviceCopy::copy(unsigned char *d_buf, const unsigned char *v, size_t nElms) const;
template void DeviceCopy::copy(short *d_buf, const short *v, size_t nElms) const;
template void DeviceCopy::copy(unsigned short *d_buf, const unsigned short *v, size_t nElms) const;
template void DeviceCopy::copy(int *d_buf, const int *v, size_t nElms) const;
template void DeviceCopy::copy(unsigned int *d_buf, const unsigned int *v, size_t nElms) const;
template void DeviceCopy::copy(long *d_buf, const long *v, size_t nElms) const;
template void DeviceCopy::copy(unsigned long *d_buf, const unsigned long *v, size_t nElms) const;
template void DeviceCopy::copy(long long *d_buf, const long long *v, size_t nElms) const;
template void DeviceCopy::copy(unsigned long long *d_buf, const unsigned long long *v, size_t nElms) const;
