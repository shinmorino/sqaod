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

    void seed(unsigned long seed);

    void setProblem(const Matrix &W, sqaod::OptimizeMethod om);

    void getProblemSize(int *N) const;
    
    void setTileSize(int tileSize);
    
    const sqaod::BitsArray &get_x() const;

    const Vector &get_E() const;

    /* executed aynchronously */
    
    void initSearch();

    void finSearch();

    void batchCalculate_E(unsigned long long iBegin, unsigned long long iEnd);

    void updateXmins();

    void search();
    
private:    
    int N_;
    Matrix W_;
    sqaod::OptimizeMethod om_;
    int tileSize_;
    unsigned long long xMax_;

    Vector E_;
    sqaod::BitsArray xList_;
    sqaod::PackedBitsArray packedXmin_;
    real Emin_;
    
    DevicePackedBitsArray d_xMin_;
    DeviceBatchSearch batchSearch_;
    
    DeviceCopy devCopy_;

    DeviceStream *stream_;
};

}

#endif

