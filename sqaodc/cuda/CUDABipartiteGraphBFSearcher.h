/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/cuda/Device.h>
#include <sqaodc/cuda/DeviceMatrix.h>
#include <sqaodc/cuda/DeviceBipartiteGraphBatchSearch.h>

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
#include <sqaodc/common/RangeMap.h>
#endif

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

    typedef sq::BitSetPairArray BitSetPairArray;
    typedef sq::PackedBitSet PackedBitSet;
    typedef sq::PackedBitSetPairArray PackedBitSetPairArray;
    typedef sq::SizeType SizeType;
    typedef sq::IdxType IdxType;
    
public:
    CUDABipartiteGraphBFSearcher();
    CUDABipartiteGraphBFSearcher(Device &device);

    ~CUDABipartiteGraphBFSearcher();

    void deallocate();
    
    void assignDevice(Device &device);
    
    /* void getProblemSize(int *N0, int *N1) const; */

    void setQUBO(const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                 sq::OptimizeMethod om = sq::optMinimize);

    /* void setPreference(const Preference &pref); */

    sq::Preferences getPreferences() const;

    const BitSetPairArray &get_x() const;

    const HostVector &get_E() const;

    void prepare();

    void calculate_E();

    void makeSolution();

    bool searchRange(PackedBitSet *curX0, PackedBitSet *curX1);

    /* void search(); */
    
private:    
    bool deviceAssigned_;
    
    HostMatrix W_;
    HostVector b0_, b1_;

    HostVector E_;
    BitSetPairArray minXPairs_;
    real Emin_;
    PackedBitSetPairArray xPackedPairs_;
    DevicePackedBitSetPairArray h_packedMinXPairs_;
    
    DeviceBatchSearch batchSearch_;
    DeviceCopy devCopy_;

#ifdef SQAODC_ENABLE_RANGE_COVERAGE_TEST
    sqaod_internal::RangeMapArray rangeMapArray_;
#endif
    typedef CUDABipartiteGraphBFSearcher<real> This;
    typedef sq::BipartiteGraphBFSearcher<real> Base;
    using Base::N0_;
    using Base::N1_;
    using Base::om_;
    using Base::tileSize0_;
    using Base::tileSize1_;
    using Base::x0_;
    using Base::x1_;
    using Base::x0max_;
    using Base::x1max_;

    /* searcher state */
    using Base::solPrepared;
    using Base::solProblemSet;
    using Base::solEAvailable;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::clearState;
    using Base::isEAvailable;
    using Base::isSolutionAvailable;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotPrepared;
};

}
