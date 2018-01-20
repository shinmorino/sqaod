#include "DeviceCopy.h"
#include "cudafuncs.h"
#include "Device.h"
#include "Assertion.h"

using namespace sqaod_cuda;
using sqaod::SizeType;
using sqaod::IdxType;



template<class real> 
DeviceCopyType<real>::DeviceCopyType() {
    devAlloc_ = NULL;
    stream_ = NULL;
}

template<class real> 
DeviceCopyType<real>::DeviceCopyType(Device &device, DeviceStream *devStream) {
    set(device, devStream);
}

template<class real> void DeviceCopyType<real>::
set(Device &device, DeviceStream *devStream) {
    devAlloc_ = device.objectAllocator<real>();
    if (devStream != NULL)
        stream_ = devStream->getCudaStream();
    else
        stream_ = NULL;
}

template<class real> void DeviceCopyType<real>::
synchronize() const {
    throwOnError(cudaStreamSynchronize(stream_));
}


template<class real> void DeviceCopyType<real>::
operator()(DeviceMatrix *dst, const HostMatrix &src) {
    devAlloc_->allocateIfNull(dst, src.dim());
    assertSameShape(*dst, src, __func__);
    copy(dst->d_data, src.data, src.rows * src.cols);
}

template<class real> void DeviceCopyType<real>::
operator()(HostMatrix *dst, const DeviceMatrix &src) const {
    if (dst->data == NULL)
        dst->resize(src.dim());
    assertSameShape(*dst, src, __func__);
    copy(dst->data, src.d_data, src.rows * src.cols);
}

template<class real> void DeviceCopyType<real>::
operator()(DeviceMatrix *dst, const DeviceMatrix &src) {
    devAlloc_->allocateIfNull(dst, src.dim());
    assertSameShape(*dst, src, __func__);
    copy(dst->d_data, src.d_data, src.rows * src.cols);
}

template<class real> void DeviceCopyType<real>::
operator()(DeviceMatrix *dst, const real &src) const {
    assertValidMatrix(*dst, __func__);
    copyBroadcast(dst->d_data, src, dst->rows * dst->cols);
}

template<class real> void DeviceCopyType<real>::
operator()(DeviceMatrix *dst, const real &src, sqaod::SizeType size,
           sqaod::SizeType stride, sqaod::IdxType offset) const {
    assertValidMatrix(*dst, __func__);
    copyBroadcastStrided(dst->d_data, src, size, stride, offset);
}
    
template<class real> void DeviceCopyType<real>::
operator()(DeviceVector *dst, const HostVector &src) {
    devAlloc_->allocateIfNull(dst, src.size);
    assertSameShape(*dst, src, __func__);
    copy(dst->d_data, src.data, src.size);
}
    
template<class real> void DeviceCopyType<real>::
operator()(HostVector *dst, const DeviceVector &src) const {
    if (dst->data == NULL)
        dst->allocate(src.size);
    assertSameShape(*dst, src, __func__);
    dst->size = src.size;
    copy(dst->data, src.d_data, src.size);
}

template<class real> void DeviceCopyType<real>::
operator()(DeviceVector *dst, const DeviceVector &src) {
    devAlloc_->allocateIfNull(dst, src.size);
    assertSameShape(*dst, src, __func__);
    copy(dst->d_data, src.d_data, src.size);
}

template<class real> void DeviceCopyType<real>::
operator()(DeviceVector *dst, const real &src) const {
    assertValidVector(*dst, __func__);
    copyBroadcast(dst->d_data, src, 1);
}

/* Host scalar variables <-> DeviceScalar */
    
template<class real> void DeviceCopyType<real>::
operator()(DeviceScalar *dst, const real &src) {
    devAlloc_->allocateIfNull(dst);
    assertValidScalar(*dst, __func__);
    copy(dst->d_data, &src, 1);
}

template<class real> void DeviceCopyType<real>::
operator()(real *dst, const DeviceScalar &src) const {
    copy(dst, src.d_data, 1);
}

/* Packed bits */
template<class real> void DeviceCopyType<real>::
operator()(sqaod::PackedBitsArray *dst, const DevicePackedBitsArray &src) const {
    assert(!"not implemented.");
    // copy(dst->data_, src.d_data, src.size);
    // dst->size_ = src.size;
}

template<class real> void DeviceCopyType<real>::
operator()(DevicePackedBitsArray *dst, const sqaod::PackedBitsArray &src) {
    assert(!"not implemented.");
    // copy(dst->data_, src.d_data, src.size);
    // dst->size_ = src.size;
}

template struct sqaod_cuda::DeviceCopyType<double>;
template struct sqaod_cuda::DeviceCopyType<float>;

