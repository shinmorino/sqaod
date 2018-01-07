#ifndef SQAOD_CUDA_DEVICE_BATCH_SEARCH_H__
#define SQAOD_CUDA_DEVICE_BATCH_SEARCH_H__

#include <cuda/CUDAFormulas.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceObjectAllocator.h>


namespace sqaod_cuda {

template<class real>
class DeviceDenseGraphBatchSearch {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceObjectAllocatorType<real> DeviceObjectAllocator;
    typedef CUDADGFuncs<real> DGFuncs;
    typedef sqaod::MatrixType<real> HostMatrix;
    
public:
    DeviceDenseGraphBatchSearch();

    void setProblem(const HostMatrix &W);

    void calculate_E(sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);

    /* sync by using a stream first, and get Emin */
    real get_Emin();
    
    void partition_xMin();

    void sync();
    
    DevicePackedBitsArray *get_xMin();
    
private:
    DeviceMatrix d_W_;
    DeviceMatrix bitsSeq_;
    DeviceMatrix Ebatch_;
    DeviceScalar d_Emin_;
    real *h_minE_;

    DGFuncs dgFuncs_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;
};

}

#endif
