/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <cuda/Device.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceDenseGraphBatchSearch.h>

namespace sqaod_cuda {

template<class real>
class CUDADenseGraphBFSearcher {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceDenseGraphBatchSearch<real> DeviceBatchSearch;
    
    typedef sqaod::MatrixType<real> Matrix;
    typedef sqaod::VectorType<real> Vector;
    
public:
    CUDADenseGraphBFSearcher();

    CUDADenseGraphBFSearcher(Device &device);

    ~CUDADenseGraphBFSearcher();

    void assignDevice(Device &device);
    
    void setProblem(const Matrix &W, sqaod::OptimizeMethod om = sqaod::optMinimize);

    void getProblemSize(sqaod::SizeType *N) const;
    
    void setTileSize(sqaod::SizeType tileSize);
    
    const sqaod::BitsArray &get_x() const;

    const Vector &get_E() const;

    /* executed aynchronously */
    
    void initSearch();

    void finSearch();

    void searchRange(sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);

    void search();
    
private:    
    int N_;
    Matrix W_;
    sqaod::OptimizeMethod om_;
    sqaod::SizeType tileSize_;
    sqaod::PackedBits xMax_;

    Vector E_;
    sqaod::BitsArray xList_;
    real Emin_;
    DevicePackedBitsArray h_packedXmin_;
    DeviceBatchSearch batchSearch_;
    DeviceCopy devCopy_;
};

}
