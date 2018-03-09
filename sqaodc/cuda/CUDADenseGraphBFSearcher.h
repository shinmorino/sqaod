/* -*- c++ -*- */
#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/cuda/Device.h>
#include <sqaodc/cuda/DeviceMatrix.h>
#include <sqaodc/cuda/DeviceDenseGraphBatchSearch.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
class CUDADenseGraphBFSearcher : public sq::DenseGraphBFSearcher<real> {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceDenseGraphBatchSearch<real> DeviceBatchSearch;
    
    typedef sq::MatrixType<real> Matrix;
    typedef sq::VectorType<real> Vector;
    
public:
    CUDADenseGraphBFSearcher();

    CUDADenseGraphBFSearcher(Device &device);
    
    ~CUDADenseGraphBFSearcher();

    void deallocate();
    
    void assignDevice(Device &device);
    
    void setProblem(const Matrix &W, sq::OptimizeMethod om = sq::optMinimize);

    /* void getProblemSize(sq::SizeType *N) const; */

    /* void setPreference(const sq::Preference &pref); */

    sq::Preferences getPreferences() const;
    
    const sq::BitsArray &get_x() const;

    const Vector &get_E() const;

    /* executed aynchronously */
    
    void prepare();

    void calculate_E();
    
    void makeSolution();

    bool searchRange(sq::PackedBits *curX);

    /* void search(); */
    
private:    
    bool deviceAssigned_;
    Matrix W_;

    Vector E_;
    sq::BitsArray xList_;
    real Emin_;
    DevicePackedBitsArray h_packedXmin_;
    DeviceBatchSearch batchSearch_;
    DeviceCopy devCopy_;

    typedef sq::DenseGraphBFSearcher<real> Base;
    using Base::N_;
    using Base::om_;
    using Base::tileSize_;
    using Base::x_;
    using Base::xMax_;

    /* searcher state */
    using Base::solPrepared;
    using Base::solProblemSet;
    using Base::solEAvailable;
    using Base::solSolutionAvailable;
    using Base::setState;
    using Base::clearState;
    using Base::isPrepared;
    using Base::throwErrorIfProblemNotSet;
    using Base::throwErrorIfNotPrepared;
    using Base::throwErrorIfENotAvailable;
    using Base::throwErrorIfSolutionNotAvailable;
};

}
