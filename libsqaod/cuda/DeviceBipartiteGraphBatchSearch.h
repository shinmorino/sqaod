#pragma once

#include <cuda/CUDAFormulas.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceObjectAllocator.h>

namespace sqaod_cuda {

class Device;

template<class real>
class DeviceBipartiteGraphBatchSearch {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceScalarType<sqaod::SizeType> DeviceSize;
    typedef CUDABGFuncs<real> BGFuncs;
    typedef sqaod::MatrixType<real> HostMatrix;
    typedef sqaod::VectorType<real> HostVector;
    
public:
    DeviceBipartiteGraphBatchSearch();

    void assignDevice(Device &device, DeviceStream *devStream);

    void deallocate();
    
    void setProblem(const HostVector &b0, const HostVector &b1,
                    const HostMatrix &W,
                    sqaod::SizeType tileSize0, sqaod::SizeType tileSize1);
    
    void calculate_E(sqaod::PackedBits xBegin0, sqaod::PackedBits xEnd0,
                     sqaod::PackedBits xBegin1, sqaod::PackedBits xEnd1);

    void partition_minXPairs(bool append);
    
    /* sync by using a stream first, and get Emin */
    real get_Emin() const {
        return *h_Emin_.d_data;
    }

    const DevicePackedBitsPairArray &get_minXPairs() const {
        return d_minXPairs_;
    }
    
    void synchronize();

    
    /* Device kernels, declared as public for tests */

    void generateBitsSequence(real *d_data, int N,
                              sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);

    void select(sqaod::PackedBitsPair *d_out, sqaod::SizeType *d_nOut,
                sqaod::PackedBits xBegin0, sqaod::PackedBits xBegin1, 
                real val, const real *d_vals, sqaod::SizeType nIn0, sqaod::SizeType nIn1);

private:
    sqaod::SizeType N0_, N1_;
    DeviceVector d_b0_;
    DeviceVector d_b1_;
    DeviceMatrix d_W_;
    sqaod::SizeType tileSize0_, tileSize1_;
    sqaod::SizeType minXPairsSize_;
    
    /* starting x of batch calculation */
    sqaod::PackedBits xBegin0_, xBegin1_;

    /* calculate_E */
    DeviceMatrix d_bitsMat0_, d_bitsMat1_;
    DeviceMatrix d_Ebatch_;
    DeviceScalar h_Emin_; /* host mem */
    /* partition */
    DevicePackedBitsPairArray d_minXPairs_;
    DeviceSize h_nMinXPairs_;
    /* lower level objects. */
    BGFuncs bgFuncs_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};

}
