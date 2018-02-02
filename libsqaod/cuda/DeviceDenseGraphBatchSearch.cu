#include "DeviceDenseGraphBatchSearch.h"
#include "Device.h"

using namespace sqaod_cuda;
namespace sq = sqaod;


template<class real>
DeviceDenseGraphBatchSearch<real>::DeviceDenseGraphBatchSearch() {
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::assignDevice(Device &device) {
    devStream_ = device.defaultStream();
    dgFuncs_.assignDevice(device, devStream_);
    devCopy_.assignDevice(device, devStream_);
    kernels_.assignStream(devStream_);
    devAlloc_ = device.objectAllocator();
}

template<class real>
void DeviceDenseGraphBatchSearch<real>::deallocate() {
    devAlloc_->deallocate(d_bitsMat_);
    devAlloc_->deallocate(d_Ebatch_);

    HostObjectAllocator halloc;
    halloc.deallocate(h_nXMins_);
    halloc.deallocate(h_Emin_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::setProblem(const HostMatrix &W, sq::SizeType tileSize) {
    devCopy_(&d_W_, W);
    tileSize_ = tileSize;
    devAlloc_->allocate(&d_bitsMat_, tileSize, W.rows);
    devAlloc_->allocate(&d_Ebatch_, tileSize);
    devAlloc_->allocate(&d_xMins_, tileSize * 2);

    HostObjectAllocator halloc;
    halloc.allocate(&h_nXMins_);
    halloc.allocate(&h_Emin_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::calculate_E(sq::PackedBits xBegin, sq::PackedBits xEnd) {
    xBegin_ = xBegin;
    sq::SizeType nBatch = sq::SizeType(xEnd - xBegin);
    abortIf(tileSize_ < nBatch,
            "nBatch is too large, tileSize=%d, nBatch=%d", int(tileSize_), int(nBatch));
    int N = d_W_.rows;
    kernels_.generateBitsSequence(d_bitsMat_.d_data, N, xBegin, xEnd);
    dgFuncs_.calculate_E(&d_Ebatch_, d_W_, d_bitsMat_);
    dgFuncs_.devMath.min(&h_Emin_, d_Ebatch_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::partition_xMins(bool append) {
    assert(d_Ebatch_.size == tileSize_);
    if (!append) {
        /* overwrite */
        d_xMins_.size = 0;
        kernels_.select(d_xMins_.d_data, h_nXMins_.d_data,
                        xBegin_, *h_Emin_.d_data, d_Ebatch_.d_data, tileSize_);
        synchronize();
        d_xMins_.size = *h_nXMins_.d_data; /* sync field */
    }
    else if (d_xMins_.size < tileSize_) {
        /* append */
        kernels_.select(&d_xMins_.d_data[d_xMins_.size], h_nXMins_.d_data,
                        xBegin_, *h_Emin_.d_data, d_Ebatch_.d_data, tileSize_);
        synchronize();
        d_xMins_.size += *h_nXMins_.d_data; /* sync field */
    }
}

template<class real>
void DeviceDenseGraphBatchSearch<real>::synchronize() {
    devStream_->synchronize();
}



template class sqaod_cuda::DeviceDenseGraphBatchSearch<double>;
template class sqaod_cuda::DeviceDenseGraphBatchSearch<float>;
