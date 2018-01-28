#ifndef SQAOD_CUDA_DEVICE_BATCH_SEARCH_H__
#define SQAOD_CUDA_DEVICE_BATCH_SEARCH_H__

#include <cuda/CUDAFormulas.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceObjectAllocator.h>
#include <cuda/DeviceAlgorithm.h>

namespace sqaod_cuda {

class Device;

template<class real>
class DeviceDenseGraphBatchSearch {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceScalarType<sqaod::SizeType> DeviceSize;
    typedef CUDADGFuncs<real> DGFuncs;
    typedef sqaod::MatrixType<real> HostMatrix;
    
public:
    DeviceDenseGraphBatchSearch();

    void assignDevice(Device &device);
    
    void setProblem(const HostMatrix &W, sqaod::SizeType tileSize);
    
    void calculate_E(sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);

    void partition_xMins(bool append);
    
    /* sync by using a stream first, and get Emin */
    real get_Emin() const {
        return *h_Emin_.d_data;
    }

    const PackedBitsArray &get_xMins() const {
        return h_xMins_;
    }
    
    void synchronize();
    
private:
    DeviceMatrix d_W_;
    sqaod::SizeType tileSize_;

    /* calculate_E */
    DeviceMatrix d_bitsMat_;
    DeviceVector d_Ebatch_;
    DeviceScalar h_Emin_; /* host mem */
    DevicePackedBitsArray d_bitsSeq_;
    /* partition */
    DevicePackedBitsArray d_xMins_;
    sqaod::PackedBitsArray h_xMins_;
    DeviceSize h_nXMins_;
    sqaod::SizeType nXMins_;
    /* lower level objects. */
    DGFuncs dgFuncs_;
    DeviceCopy devCopy_;
    DeviceAlgorithm devAlgo_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};

}

#endif
