#include "utils.h"
#include <cuda/DeviceCopy.h>
#include <common/EigenBridge.h>


template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceMatrixType<real> &dmat) {
    sqaod::MatrixType<real> hmat;
    DeviceCopy()(&hmat, dmat);
    ostm << sq::mapTo(hmat) << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceVectorType<real> &dvec) {
    sqaod::VectorType<real> hvec;
    DeviceCopy()(&hvec, dvec);
    ostm << mapToRowVector(hvec) << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceScalarType<real> &ds) {
    real hs;
    DeviceCopy()(&hs, ds);
    ostm << hs << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const sq::ArrayType<real> &arr) {
    for (int idx = 0; idx < (int)arr.size(); ++idx)
        ostm << arr[idx] << ", ";
    ostm << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const sq::VectorType<real> &vec) {
    for (int idx = 0; idx < (int)vec.size; ++idx)
        ostm << vec.data[idx] << ", "; /* FIXME: add [] operator */
    ostm << std::endl;
    return ostm;
}

template<class real>
void show(const sqaod_cuda::DeviceMatrixType<real> &dmat, const sqaod::MatrixType<real> &hmat) {
    std::cerr << std::endl 
        << "Device" << std::endl
        << dmat
        << "Host  " << std::endl
        << sq::mapTo(hmat) << std::endl; 
}

template<class real>
void show(const sqaod_cuda::DeviceVectorType<real> &dvec, const sqaod::VectorType<real> &hvec) {
    std::cerr << std::endl
        << "Device" << std::endl
        << dvec
        << "Host  " << std::endl
        << mapToRowVector(hvec) << std::endl; 
}




template std::ostream &operator<<(std::ostream &ostm, const DeviceMatrixType<double> &dmat);
template std::ostream &operator<<(std::ostream &ostm, const DeviceVectorType<double> &dvec);
template std::ostream &operator<<(std::ostream &ostm, const DeviceScalarType<double> &ds);
template std::ostream &operator<<(std::ostream &ostm, const sq::ArrayType<double> &arr);
template std::ostream &operator<<(std::ostream &ostm, const sq::VectorType<double> &vec);
template void show(const sqaod_cuda::DeviceMatrixType<double> &dmat, const sqaod::MatrixType<double> &hmat);
template void show(const sqaod_cuda::DeviceVectorType<double> &dvec, const sqaod::VectorType<double> &hvec);

template std::ostream &operator<<(std::ostream &ostm, const DeviceMatrixType<float> &dmat);
template std::ostream &operator<<(std::ostream &ostm, const DeviceVectorType<float> &dvec);
template std::ostream &operator<<(std::ostream &ostm, const DeviceScalarType<float> &ds);
template std::ostream &operator<<(std::ostream &ostm, const sq::ArrayType<float> &arr);
template std::ostream &operator<<(std::ostream &ostm, const sq::VectorType<float> &vec);
template void show(const sqaod_cuda::DeviceMatrixType<float> &dmat, const sqaod::MatrixType<float> &hmat);
template void show(const sqaod_cuda::DeviceVectorType<float> &dvec, const sqaod::VectorType<float> &hvec);


template<class real>
sq::VectorType<real> segmentedSum(const sq::MatrixType<real> &A, sq::SizeType segLen, sq::SizeType nSegments) {
    sq::VectorType<real> x;
    sq::EigenMatrixType<real> eA = sq::mapTo(A);
    x.resize(nSegments);
    sq::mapToRowVector(x) = eA.rowwise().sum();
    return x;
}

template sq::VectorType<double> segmentedSum(const sq::MatrixType<double> &A, sq::SizeType segLen, sq::SizeType nSegments);
template sq::VectorType<float> segmentedSum(const sq::MatrixType<float> &A, sq::SizeType segLen, sq::SizeType nSegments);


template<class real>
bool allclose(const DeviceVectorType<real> &dvec, const sqaod::VectorType<real> &hvec, real epsiron) {
    sqaod::VectorType<real> copied;
    DeviceCopy devCopy;
    devCopy(&copied, dvec);
    devCopy.synchronize();
    real absDiff = (sq::mapToRowVector(copied) - sq::mapToRowVector(hvec)).array().abs().sum();
    real absVal = sq::mapToRowVector(copied).array().abs().sum() + sq::mapToRowVector(hvec).array().abs().sum();
    // fprintf(stderr, "%g %g\n", absDiff, absVal);
    if (absDiff < epsiron)
        return true;
    return ((absDiff / absVal) < epsiron);
}

template bool allclose(const DeviceVectorType<float> &dvec, const sqaod::VectorType<float> &hvec, float epsiron);
template bool allclose(const DeviceVectorType<double> &dvec, const sqaod::VectorType<double> &hvec, double epsiron);
