#include "DeviceDenseGraphBatchSearch.h"

using namespace sqaod_cuda;

template<class real>
void DeviceDenseGraphBatchSearch<real>::setProblem(const HostMatrix &W) {
    device_->allocate(&d_W_, W.dim());
    devCopy_(d_W_, W);
}

template<class real>
void DeviceDenseGraphBatchSearch<real>::calculate_E(sqaod::PackedBits xBegin, sqaod::PackedBits xEnd) {
    int nBatch = int(xEnd - xBegin);
    int N = d_W_.rows;

    DeviceMatrix bitsSeq(nBatch, N);
    DeviceMatrix Ebatch(nBatch, 1);
    createBitsSequence(bitsSeq.data(), N, xBegin, xEnd);
    dgFuncs_.calculate_E(&Ebatch, d_W_, bitsSeq);
    dgFuncs_.devMath.min(&d_Emin_, Ebatch);
}
