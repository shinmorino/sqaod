#pragma once

#include <cuda/Device.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceFormulas.h>

namespace sqaod_cuda {

namespace sq = sqaod;

class Device;

template<class real>
class DeviceBipartiteGraphBatchSearch {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceScalarType<sq::SizeType> DeviceSize;
    typedef DeviceBipartiteGraphFormulas<real> DeviceFormulas;
    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    
public:
    DeviceBipartiteGraphBatchSearch();

    void assignDevice(Device &device, DeviceStream *devStream);

    void deallocate();
    
    void setProblem(const HostVector &b0, const HostVector &b1,
                    const HostMatrix &W,
                    sq::SizeType tileSize0, sq::SizeType tileSize1);
    
    void calculate_E(sq::PackedBits xBegin0, sq::PackedBits xEnd0,
                     sq::PackedBits xBegin1, sq::PackedBits xEnd1);

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
                              sq::PackedBits xBegin, sq::PackedBits xEnd);

    void select(sq::PackedBitsPair *d_out, sq::SizeType *d_nOut,
                sq::PackedBits xBegin0, sq::PackedBits xBegin1, 
                real val, const real *d_vals, sq::SizeType nIn0, sq::SizeType nIn1);

private:
    sq::SizeType N0_, N1_;
    DeviceVector d_b0_;
    DeviceVector d_b1_;
    DeviceMatrix d_W_;
    sq::SizeType tileSize0_, tileSize1_;
    sq::SizeType minXPairsSize_;
    
    /* starting x of batch calculation */
    sq::PackedBits xBegin0_, xBegin1_;

    /* calculate_E */
    DeviceMatrix d_bitsMat0_, d_bitsMat1_;
    DeviceMatrix d_Ebatch_;
    DeviceScalar h_Emin_; /* host mem */
    /* partition */
    DevicePackedBitsPairArray d_minXPairs_;
    DeviceSize h_nMinXPairs_;
    /* lower level objects. */
    DeviceFormulas devFormulas_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};

}
