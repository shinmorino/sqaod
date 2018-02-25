#pragma once

#include <cuda/DeviceFormulas.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceObjectAllocator.h>

namespace sqaod_cuda {

namespace sq = sqaod;

class Device;

template<class real>
class DeviceDenseGraphBatchSearch {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceScalarType<sq::SizeType> DeviceSize;
    typedef DeviceDenseGraphFormulas<real> Formulas;
    typedef sq::MatrixType<real> HostMatrix;
    
public:
    DeviceDenseGraphBatchSearch();

    void assignDevice(Device &device);

    void deallocate();
    
    void setProblem(const HostMatrix &W, sq::SizeType tileSize);
    
    void calculate_E(sq::PackedBits xBegin, sq::PackedBits xEnd);

    void partition_xMins(bool append);
    
    /* sync by using a stream first, and get Emin */
    real get_Emin() const {
        return *h_Emin_.d_data;
    }

    const DevicePackedBitsArray &get_xMins() const {
        return d_xMins_;
    }
    
    void synchronize();


    /* Device kernels, declared as public for tests */

    void generateBitsSequence(real *d_data, int N,
                              sq::PackedBits xBegin, sq::PackedBits xEnd);

    void select(sq::PackedBits *d_out, sq::SizeType *d_nOut, sq::PackedBits xBegin, 
                real val, const real *d_vals, sq::SizeType nIn);

private:
    sq::SizeType N_;
    DeviceMatrix d_W_;
    sq::SizeType tileSize_;

    /* starting x of batch calculation */
    sq::PackedBits xBegin_;

    /* calculate_E */
    DeviceMatrix d_bitsMat_;
    DeviceVector d_Ebatch_;
    DeviceScalar h_Emin_; /* host mem */
    /* partition */
    DevicePackedBitsArray d_xMins_;
    DeviceSize h_nXMins_;
    /* lower level objects. */
    Formulas devFormulas_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};

}
