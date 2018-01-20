#pragma once

#include <common/Matrix.h>
#include <iostream>


template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceMatrixType<real> &dmat) {
    sqaod::MatrixType<real> hmat;
    DeviceCopyType<real>()(&hmat, dmat);
    ostm << hmat.map() << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceVectorType<real> &dvec) {
    sqaod::VectorType<real> hvec;
    DeviceCopyType<real>()(&hvec, dvec);
    ostm << hvec.mapToRowVector() << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceScalarType<real> &ds) {
    real hs;
    DeviceCopyType<real>()(&hs, ds);
    ostm << hs << std::endl;
    return ostm;
}

template<class real>
void show(const sqaod_cuda::DeviceMatrixType<real> &dmat, const sqaod::MatrixType<real> &hmat) {
    std::cerr << std::endl 
        << "Device" << std::endl
        << dmat
        << "Host  " << std::endl
        << hmat.map() << std::endl; 
}

template<class real>
void show(const sqaod_cuda::DeviceVectorType<real> &dvec, const sqaod::VectorType<real> &hvec) {
    std::cerr << std::endl
        << "Device" << std::endl
        << dvec
        << "Host  " << std::endl
        << hvec.mapToRowVector() << std::endl; 
}


template<class real>
bool operator==(const DeviceMatrixType<real> &dmat, const sqaod::MatrixType<real> &hmat) {
    sqaod::MatrixType<real> copied;
    DeviceCopyType<real> devCopy;
    devCopy(&copied, dmat);
    devCopy.synchronize();
    return copied.map() == hmat.map();
}

template<class real>
bool operator==(const DeviceVectorType<real> &dvec, const sqaod::VectorType<real> &hvec) {
    sqaod::VectorType<real> copied;
    DeviceCopyType<real> devCopy;
    devCopy(&copied, dvec);
    devCopy.synchronize();
    return copied.mapToRowVector() == hvec.mapToRowVector();
}

template<class real>
bool operator==(const DeviceScalarType<real> &dsc, const real &hsc) {
    real copied;
    DeviceCopyType<real> devCopy;
    devCopy(&copied, dsc);
    devCopy.synchronize();
    return copied == hsc;
}


template<class real>
sqaod::MatrixType<real> testMat(const sqaod::Dim &dim) {
    sqaod::MatrixType<real> hmat(dim);
    for (sqaod::SizeType iRow = 0; iRow < dim.rows; ++iRow) {
        for (sqaod::SizeType iCol = 0; iCol < dim.cols; ++iCol) {
            hmat(iRow, iCol) = real(iRow * 10 + iCol);
        }
    }
    return hmat;
}

template<class real>
sqaod::VectorType<real> testVec(const sqaod::SizeType size) {
    sqaod::VectorType<real> hvec(size);
    for (sqaod::SizeType idx = 0; idx < size; ++idx) {
        hvec(idx) = real((idx * 3) % 17);
    }
    return hvec;
}

template<class real>
sqaod::MatrixType<real> testMatBalanced(const sqaod::Dim &dim) {
    int v = -2;
    sqaod::MatrixType<real> hmat(dim);
    for (sqaod::SizeType iRow = 0; iRow < dim.rows; ++iRow) {
        for (sqaod::SizeType iCol = 0; iCol < dim.cols; ++iCol) {
            hmat(iRow, iCol) = (real)v;
            if (++v == 3)
                v = -2;
        }
    }
    return hmat;
}

template<class real>
sqaod::VectorType<real> testVecBalanced(const sqaod::SizeType size) {
    int v = -2;
    sqaod::VectorType<real> hvec(size);
    for (sqaod::SizeType idx = 0; idx < size; ++idx) {
        hvec(idx) = (real)v;
        if (++v == 3)
            v = -2;
    }
    return hvec;
}

