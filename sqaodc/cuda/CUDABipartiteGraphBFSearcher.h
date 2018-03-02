/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/cuda/Device.h>
#include <sqaodc/cuda/DeviceMatrix.h>
#include <sqaodc/cuda/DeviceBipartiteGraphBatchSearch.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
class CUDABipartiteGraphBFSearcher : public sq::BipartiteGraphBFSearcher<real> {
    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceBipartiteGraphBatchSearch<real> DeviceBatchSearch;
    typedef DeviceMathType<real> DeviceMath;

    typedef sq::BitsPairArray BitsPairArray;
    typedef sq::PackedBits PackedBits;
    typedef sq::PackedBitsPairArray PackedBitsPairArray;
    typedef sq::SizeType SizeType;
    typedef sq::IdxType IdxType;
    
public:
    CUDABipartiteGraphBFSearcher();
    CUDABipartiteGraphBFSearcher(Device &device);

    ~CUDABipartiteGraphBFSearcher();

    void deallocate();
    
    void assignDevice(Device &device);
    
    /* void getProblemSize(int *N0, int *N1) const; */

    void setProblem(const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                    sq::OptimizeMethod om = sq::optMinimize);

    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;

    const BitsPairArray &get_x() const;

    const HostVector &get_E() const;

    void initSearch();

    void finSearch();

    void searchRange(PackedBits x0Begin, PackedBits x0End,
                     PackedBits x1Begin, PackedBits x1End);

    /* void search(); */
    
private:    
    bool deviceAssigned_;
    
    HostMatrix W_;
    HostVector b0_, b1_;

    HostVector E_;
    BitsPairArray minXPairs_;
    real Emin_;
    PackedBitsPairArray xPackedPairs_;
    DevicePackedBitsPairArray h_packedMinXPairs_;
    
    DeviceBatchSearch batchSearch_;
    DeviceCopy devCopy_;

    typedef sq::BipartiteGraphBFSearcher<real> Base;
    using Base::N0_;
    using Base::N1_;
    using Base::om_;
    using Base::tileSize0_;
    using Base::tileSize1_;
    using Base::x0max_;
    using Base::x1max_;

    /* searcher state */
    using Base::solInitialized;
    using Base::solProblemSet;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::isInitialized;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotInitialized;
    using Base::throwErrorIfSolutionNotAvailable;
};

}
