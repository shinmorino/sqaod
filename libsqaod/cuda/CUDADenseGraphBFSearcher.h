/* -*- c++ -*- */
#pragma once

#include <common/Common.h>
#include <cuda/Device.h>
#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceDenseGraphBatchSearch.h>

namespace sqaod_cuda {

template<class real>
class CUDADenseGraphBFSearcher : public sqaod::DenseGraphBFSearcher<real> {
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

    void deallocate();
    
    void assignDevice(Device &device);
    
    void setProblem(const Matrix &W, sqaod::OptimizeMethod om = sqaod::optMinimize);

    /* void getProblemSize(sqaod::SizeType *N) const; */

    /* void setPreference(const sqaod::Preference &pref); */

    /* sqaod::Preferences getPreferences() const; */
    
    const sqaod::BitsArray &get_x() const;

    const Vector &get_E() const;

    /* executed aynchronously */
    
    void initSearch();

    void finSearch();

    void searchRange(sqaod::PackedBits xBegin, sqaod::PackedBits xEnd);

    /* void search(); */
    
private:    
    bool deviceAssigned_;
    Matrix W_;

    Vector E_;
    sqaod::BitsArray xList_;
    real Emin_;
    DevicePackedBitsArray h_packedXmin_;
    DeviceBatchSearch batchSearch_;
    DeviceCopy devCopy_;

    typedef sqaod::DenseGraphBFSearcher<real> Base;
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
