#include "DeviceCopy.h"
#include "cudafuncs.h"

using namespace sqaod_cuda;
using sqaod::SizeType;
using sqaod::IdxType;

template<class V>
void DeviceCopy::copy(V *d_buf, const V *v, SizeType nElms) const {
    throwOnError(cudaMemcpyAsync(d_buf, v, sizeof(V) * nElms, cudaMemcpyDefault, stream_));
}


template<class V>
__global__ static
void copyBroadcastStridedKernel(V *d_buf, const V v, SizeType size, SizeType stride, IdxType offset) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        IdxType pos = gid * stride + offset;
        d_buf[pos] = v;
    }
}



template<class V> void DeviceCopy::
copyBroadcastStrided(V *d_buf, const V &v, SizeType size, SizeType stride, IdxType offset) const {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    copyBroadcastStridedKernel<<<gridDim, blockDim>>>(d_buf, v, size, stride, offset);
    DEBUG_SYNC;
}


DeviceCopy::DeviceCopy(DeviceStream *devStream) {
    setDeviceStream(devStream);
}

void DeviceCopy::setDeviceStream(DeviceStream *devStream) {
    if (devStream != NULL)
        stream_ = devStream->getStream();
    else
        stream_ = NULL; /* use NULL stream */
}




template void DeviceCopy::copy(double *, const double *, SizeType) const;
template void DeviceCopy::copy(float *, const float *, SizeType) const;
template void DeviceCopy::copy(char *, const char *, SizeType) const;
template void DeviceCopy::copy(unsigned char *, const unsigned char *, SizeType) const;
template void DeviceCopy::copy(short *, const short *, SizeType) const;
template void DeviceCopy::copy(unsigned short *, const unsigned short *, SizeType) const;
template void DeviceCopy::copy(int *, const int *, SizeType) const;
template void DeviceCopy::copy(unsigned int *, const unsigned int *, SizeType) const;
template void DeviceCopy::copy(long *, const long *, SizeType) const;
template void DeviceCopy::copy(unsigned long *, const unsigned long *, SizeType) const;
template void DeviceCopy::copy(long long *, const long long *, SizeType) const;
template void DeviceCopy::copy(unsigned long long *, const unsigned long long *, SizeType) const;

template void DeviceCopy::copyBroadcastStrided(double *, const double &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(float *, const float &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(char *, const char &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(unsigned char *, const unsigned char &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(short *, const short &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(unsigned short *, const unsigned short &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(int *, const int &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(unsigned int *, const unsigned int &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(long *, const long &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(unsigned long *, const unsigned long &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(long long *, const long long &, SizeType, SizeType, IdxType) const;
template void DeviceCopy::copyBroadcastStrided(unsigned long long *, const unsigned long long &, SizeType, SizeType, IdxType) const;
