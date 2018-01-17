#ifndef SQAOD_CUDA_COPY_H__
#define SQAOD_CUDA_COPY_H__

#include <cuda/cudafuncs.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceStream.h>
#include <cuda/DeviceKernels.h>
#include <cuda/DeviceObjectAllocator.h>

namespace sqaod_cuda {

template<class real>
struct DeviceCopyType {

    typedef sqaod::MatrixType<real> HostMatrix;
    typedef sqaod::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
 
    template<class V>
    void copy(V *d_buf, const V *v, sqaod::SizeType nElms) const;

    template<class V>
    void copyBroadcast(V *d_buf, const V &v, sqaod::SizeType nElms) const;

    void copyBroadcastStrided(real *d_buf, const real &v, sqaod::SizeType size,
                              sqaod::SizeType stride, sqaod::IdxType offset) const;

    /* HostMatrix <-> DeviceMatrix */
    
    void operator()(DeviceMatrix *dst, const HostMatrix &src);
    
    void operator()(HostMatrix *dst, const DeviceMatrix &src) const;
    
    void operator()(DeviceMatrix *dst, const DeviceMatrix &src);
    
    void operator()(DeviceMatrix *dst, const real &src) const;
    
    void operator()(DeviceMatrix *dst, const real &src, sqaod::SizeType size,
                    sqaod::SizeType stride, sqaod::IdxType offset) const;
    
    /* HostVector <-> DeviceVector */
    
    void operator()(DeviceVector *dst, const HostVector &src);
    
    void operator()(HostVector *dst, const DeviceVector &src) const;
    
    void operator()(DeviceVector *dst, const DeviceVector &src);
    
    void operator()(DeviceVector *dst, const real &src) const;

    /* Host scalar variables <-> DeviceScalar */
    
    void operator()(DeviceScalar *dst, const real &src);
    
    void operator()(real *dst, const DeviceScalar &src) const;
    
    /* Packed bits */
    void operator()(DevicePackedBitsArray *dst, const sqaod::PackedBitsArray &src);

    void operator()(sqaod::PackedBitsArray *dst, const DevicePackedBitsArray &src) const;

    DeviceCopyType();

    DeviceCopyType(Device &device, DeviceStream *stream = NULL);
    
    void set(Device &device, DeviceStream *stream = NULL);
    
private:
    typedef DeviceObjectAllocatorType<real> DeviceObjectAllocator;
    DeviceObjectAllocator *devAlloc_;
    DeviceCopyKernels kernels_;
    cudaStream_t stream_;
};


template<class real> template<class V> inline
void DeviceCopyType<real>::copy(V *d_buf, const V *v, sqaod::SizeType nElms) const {
    throwOnError(cudaMemcpyAsync(d_buf, v, sizeof(V) * nElms, cudaMemcpyDefault, stream_));
}

template<class real> template<class V> inline
void DeviceCopyType<real>::copyBroadcast(V *d_buf, const V &v, sqaod::SizeType size) const {
    kernels_.copyBroadcast(d_buf, v, size);
}


}


#endif
