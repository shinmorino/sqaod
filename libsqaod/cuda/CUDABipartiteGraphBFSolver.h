/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/Device.h>
#include <cuda/DeviceBipartiteGraphBatchSearch.h>

namespace sqaod_cuda {

template<class real>
class CUDABipartiteGraphBFSolver {
    typedef sqaod::MatrixType<real> HostMatrix;
    typedef sqaod::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceBipartiteGraphBatchSearch<real> DeviceBatchSearch;
    typedef DeviceMathType<real> DeviceMath;

    typedef sqaod::BitsPairArray BitsPairArray;
    typedef sqaod::PackedBits PackedBits;
    typedef sqaod::PackedBitsPairArray PackedBitsPairArray;
    typedef sqaod::SizeType SizeType;
    typedef sqaod::IdxType IdxType;
public:
    CUDABipartiteGraphBFSolver();
    CUDABipartiteGraphBFSolver(Device &device, DeviceStream *devStream = NULL);

    ~CUDABipartiteGraphBFSolver();

    void assignDevice(Device &device, DeviceStream *devStream = NULL);
    
    void getProblemSize(int *N0, int *N1) const;

    void setProblem(const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                    sqaod::OptimizeMethod om);

    void setTileSize(sqaod::SizeType tileSize0, sqaod::SizeType tileSize1);

    const sqaod::BitsPairArray &get_x() const;

    const HostVector &get_E() const;

    void initSearch();

    void finSearch();

    void searchRange(PackedBits iBegin0, PackedBits iEnd0,
                     PackedBits iBegin1, PackedBits iEnd1);

    void search();
    
private:    
    SizeType N0_, N1_;
    HostMatrix W_;
    HostVector b0_, b1_;
    sqaod::OptimizeMethod om_;
    SizeType tileSize0_;
    SizeType tileSize1_;
    PackedBits x0max_;
    PackedBits x1max_;

    HostVector E_;
    BitsPairArray minXPairs_;
    real Emin_;
    PackedBitsPairArray xPackedPairs_;
    DevicePackedBitsPairArray h_packedMinXPairs_;
    
    DeviceBatchSearch batchSearch_;
    DeviceCopy devCopy_;
};

}
