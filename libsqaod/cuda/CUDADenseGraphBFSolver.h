/* -*- c++ -*- */
#ifndef SQAOD_CUDA_DENSE_GRAPH_BF_SOLVER_H__
#define SQAOD_CUDA_DENSE_GRAPH_BF_SOLVER_H__

#include <common/Common.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceStream.h>
#include <cuda/DeviceCopy.h>
#include <cuda/DeviceMath.h>
#include <cuda/DeviceRandom.h>
#include <cuda/DeviceAlgorithm.h>
#include <cuda/DeviceDenseGraphBatchSearch.h>

namespace sqaod_cuda {

template<class real>
class CUDADenseGraphBFSolver {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceMathType<real> DeviceMath;
    typedef DeviceDenseGraphBatchSearch<real> DeviceBatchSearch;
    
    typedef sqaod::MatrixType<real> Matrix;
    typedef sqaod::VectorType<real> Vector;
    
public:
    CUDADenseGraphBFSolver();
    ~CUDADenseGraphBFSolver();

    void assignDevice(Device &device);
    
    void setProblem(const Matrix &W, sqaod::OptimizeMethod om);

    void getProblemSize(sqaod::SizeType *N) const;
    
    void setTileSize(sqaod::SizeType tileSize);
    
    const sqaod::BitsArray &get_x() const;

    const Vector &get_E() const;

    /* executed aynchronously */
    
    void initSearch();

    void finSearch();

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
    
    DeviceBatchSearch batchSearch_;
};

}

#endif
