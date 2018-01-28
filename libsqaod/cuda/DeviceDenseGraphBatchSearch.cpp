#include "DeviceDenseGraphBatchSearch.h"
#include "DeviceAlgorithm.h"
#include "Device.h"

using namespace sqaod_cuda;


template<class real>
DeviceDenseGraphBatchSearch<real>::DeviceDenseGraphBatchSearch() {
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::assignDevice(Device &device) {
    devStream_ = device.defaultStream();
    dgFuncs_.assignDevice(device, devStream_);
    devCopy_.assignDevice(device, devStream_);
    devAlgo_.assignDevice(device, devStream_);
    devAlloc_ = device.objectAllocator();
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::setProblem(const HostMatrix &W, sqaod::SizeType tileSize) {
    devCopy_(&d_W_, W);
    tileSize_ = tileSize;
    devAlloc_->allocate(&d_bitsSeq_, tileSize);
    devAlloc_->allocate(&d_Ebatch_, tileSize);

    HostObjectAllocator halloc;
    halloc.allocate(&h_nXMins_);
    halloc.allocate(&h_Emin_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::calculate_E(sqaod::PackedBits xBegin, sqaod::PackedBits xEnd) {
    sqaod::SizeType nBatch = sqaod::SizeType(xEnd - xBegin);
    abortIf(tileSize_ < nBatch,
            "nBatch is too large, tileSize=%d, nBatch=%d", int(tileSize_), int(nBatch));
    int N = d_W_.rows;
    devAlgo_.generateBitsSequence(d_bitsMat_.d_data, N, xBegin, xEnd);
    dgFuncs_.calculate_E(&d_Ebatch_, d_W_, d_bitsMat_);
    dgFuncs_.devMath.min(&h_Emin_, d_Ebatch_);
}


template<class real>
void DeviceDenseGraphBatchSearch<real>::partition_xMins(bool append) {
    assert(d_Ebatch_.size == d_bitsSeq_.size);
    if (!append) {
        /* overwrite */
        nXMins_ = 0;
        devAlgo_.partition_Emin(d_xMins_.d_data, h_nXMins_.d_data,
                                *h_Emin_.d_data, d_Ebatch_.d_data,
                                d_bitsSeq_.d_data, d_bitsSeq_.size);
    }
    else if (nXMins_ <= tileSize_) {
        /* append */
        devAlgo_.partition_Emin(&d_xMins_.d_data[nXMins_], h_nXMins_.d_data,
                                *h_Emin_.d_data,
                                d_Ebatch_.d_data, d_bitsSeq_.d_data, d_bitsSeq_.size);
    }
}

template<class real>
void DeviceDenseGraphBatchSearch<real>::synchronize() {
    devStream_->synchronize();
    d_xMins_.size = *h_nXMins_.d_data; /* sync field */
    nXMins_ += *h_nXMins_.d_data;
}



template class sqaod_cuda::DeviceDenseGraphBatchSearch<double>;
template class sqaod_cuda::DeviceDenseGraphBatchSearch<float>;
