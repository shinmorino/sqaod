#pragma once

#include <sqaodc/cuda/DeviceFormulas.h>
#include <sqaodc/cuda/DeviceMatrix.h>
#include <sqaodc/cuda/DeviceArray.h>
#include <sqaodc/cuda/DeviceObjectAllocator.h>

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
    
    void setQUBO(const HostMatrix &W, sq::SizeType tileSize);
    
    void calculate_E(sq::PackedBitSet xBegin, sq::PackedBitSet xEnd);

    void partition_xMins(bool append);
    
    /* sync by using a stream first, and get Emin */
    real get_Emin() const {
        return *h_Emin_.d_data;
    }

    const DevicePackedBitSetArray &get_xMins() const {
        return d_xMins_;
    }
    
    void synchronize();


    /* Device kernels, declared as public for tests */

    void generateBitsSequence(DeviceMatrix *bitsSequences,
                              sq::PackedBitSet xBegin, sq::PackedBitSet xEnd);

    void select(sq::PackedBitSet *d_out, sq::SizeType *d_nOut, sq::PackedBitSet xBegin, 
                real val, const real *d_vals, sq::SizeType nIn);

private:
    sq::SizeType N_;
    DeviceMatrix d_W_;
    sq::SizeType tileSize_;

    /* starting x of batch calculation */
    sq::PackedBitSet xBegin_;

    /* calculate_E */
    DeviceMatrix d_bitsMat_;
    DeviceVector d_Ebatch_;
    DeviceScalar h_Emin_; /* host mem */
    /* partition */
    DevicePackedBitSetArray d_xMins_;
    DeviceSize h_nXMins_;
    /* lower level objects. */
    Formulas devFormulas_;
    DeviceCopy devCopy_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};

}
