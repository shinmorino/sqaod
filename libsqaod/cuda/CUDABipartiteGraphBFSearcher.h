/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <cuda/Device.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceBipartiteGraphBatchSearch.h>

namespace sqaod_cuda {

template<class real>
class CUDABipartiteGraphBFSearcher : public BipartiteGraphBFSearcher<real> {
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

    typedef BipartiteGraphBFSearcher<real> Base;
    using Base::N0_;
    using Base::N1_;
    using Base::om_;
    using Base::tileSize0_;
    using Base::tileSize1_;
    using Base::x0max_;
    using Base::x1max_;

public:
    CUDABipartiteGraphBFSearcher();
    CUDABipartiteGraphBFSearcher(Device &device);

    ~CUDABipartiteGraphBFSearcher();

    void assignDevice(Device &device);
    
    /* void getProblemSize(int *N0, int *N1) const; */

    void setProblem(const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                    sqaod::OptimizeMethod om = sqaod::optMinimize);

    /* void setPreference(const Preference &pref); */

    /* Preferences getPreferences() const; */

    const BitsPairArray &get_x() const;

    const HostVector &get_E() const;

    void initSearch();

    void finSearch();

    void searchRange(PackedBits x0Begin, PackedBits x0End,
                     PackedBits x1Begin, PackedBits x1End);

    /* void search(); */
    
private:    
    HostMatrix W_;
    HostVector b0_, b1_;

    HostVector E_;
    BitsPairArray minXPairs_;
    real Emin_;
    PackedBitsPairArray xPackedPairs_;
    DevicePackedBitsPairArray h_packedMinXPairs_;
    
    DeviceBatchSearch batchSearch_;
    DeviceCopy devCopy_;
};

}
