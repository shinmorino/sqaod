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

    /* sq::Preferences getPreferences() const; */
    
    const sq::BitsArray &get_x() const;

    const Vector &get_E() const;

    /* executed aynchronously */
    
    void initSearch();

    void finSearch();

    void searchRange(sq::PackedBits xBegin, sq::PackedBits xEnd);

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
    using Base::xMax_;

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
