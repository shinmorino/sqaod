#pragma once

#include <iostream>
#include <common/Matrix.h>
#include <common/Random.h>

namespace sq = sqaod;

template<class real>
real epusiron();

template<> inline
float epusiron<float>() {
    return 1.e-5f;
}

template<> inline
double epusiron<double>() {
    return 1.e-10;
}


template<class real>
std::ostream &operator<<(std::ostream &ostm, const sq::ArrayType<real> &arr);
template<class real>
std::ostream &operator<<(std::ostream &ostm, const sq::VectorType<real> &vec);


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


template<class real>
sqaod::MatrixType<real> testMatSymmetric(const sq::SizeType dim) {
    sq::MatrixType<real> mat(dim, dim);

    int v = -2;

    for (sq::SizeType irow = 0; irow < dim; ++irow) {
        for (sq::SizeType icol = irow; icol < dim; ++icol) {
            mat(icol, irow) = mat(irow, icol) = (real)v;
            if (++v == 3)
                v = -2;
        }
    }

    return mat;
}

template<class real>
sqaod::MatrixType<real> createRandomSymmetricMatrix(const sq::SizeType dim) {
    sq::MatrixType<real> mat(dim, dim);

    sq::Random random;
    random.seed(1);
    for (sq::SizeType irow = 0; irow < dim; ++irow) {
        for (sq::SizeType icol = irow; icol < dim; ++icol) {
            mat(icol, irow) = mat(irow, icol) = random.random<real>();
        }
    }

    return mat;
}

template<class real>
sqaod::VectorType<real> randomizeBits(const sq::SizeType size) {
    sq::VectorType<real> vec(size);

    sq::Random random;
    random.seed(2);
    for (sq::SizeType idx = 0; idx < size; ++idx) {
        vec(idx) = (real)random.randInt(2);
    }

    return vec;
}

template<class real>
sqaod::MatrixType<real> randomizeBits(const sq::Dim &dim) {
    sq::MatrixType<real> mat(dim);

    sq::Random random;
    random.seed(2);
    for (sq::SizeType irow = 0; irow < dim.rows; ++irow) {
        for (sq::SizeType icol = 0; icol < dim.cols; ++icol) {
            mat(irow, icol) = (real)random.randInt(2);
        }
    }
    return mat;
}

template<class real>
sq::VectorType<real> segmentedSum(const sq::MatrixType<real> &A, sq::SizeType segLen, sq::SizeType nSegments);




#ifdef SQAODC_CUDA_ENABLED

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceArray.h>
#include <cuda/DeviceCopy.h>

namespace sqcu = sqaod_cuda;

template<class real>
std::ostream &operator<<(std::ostream &ostm, const sqcu::DeviceMatrixType<real> &dmat);
template<class real>
std::ostream &operator<<(std::ostream &ostm, const sqcu::DeviceVectorType<real> &dvec);
template<class real>
std::ostream &operator<<(std::ostream &ostm, const sqcu::DeviceScalarType<real> &ds);
template<class real>
void show(const sqaod_cuda::DeviceMatrixType<real> &dmat, const sqaod::MatrixType<real> &hmat);

template<class real>
void show(const sqaod_cuda::DeviceVectorType<real> &dvec, const sqaod::VectorType<real> &hvec);


template<class real>
bool operator==(const sqcu::DeviceMatrixType<real> &dmat, const sqaod::MatrixType<real> &hmat) {
    sqaod::MatrixType<real> copied;
    sqcu::DeviceCopy devCopy;
    devCopy(&copied, dmat);
    devCopy.synchronize();
    return copied == hmat;
}

template<class real>
bool operator==(const sqcu::DeviceVectorType<real> &dvec, const sqaod::VectorType<real> &hvec) {
    sqaod::VectorType<real> copied;
    sqcu::DeviceCopy devCopy;
    devCopy(&copied, dvec);
    devCopy.synchronize();
    return copied == hvec;
}

template<class real>
bool operator==(const sqcu::DeviceScalarType<real> &dsc, const real &hsc) {
    real copied;
    sqcu::DeviceCopy devCopy;
    devCopy(&copied, dsc);
    devCopy.synchronize();
    return copied == hsc;
}

template<class real>
bool operator==(const sqcu::DeviceArrayType<real> &dsc, const sq::ArrayType<real> &hsc) {
    sqcu::DeviceArrayType<real> copied;
    HostObjectAllocator().allocate(&copied, dsc.size);
    sqcu::DeviceCopy devCopy;
    devCopy(&copied, dsc);
    devCopy.synchronize();
    if (copied.size != hsc.size()) {
        HostObjectAllocator().deallocate(copied);
        return false;
    }
    for (int idx = 0; idx < (int)copied.size; ++idx) {
        if (copied[idx] != hsc[idx]) {
            HostObjectAllocator().deallocate(copied);
            return false;
        }
    }
    HostObjectAllocator().deallocate(copied);
    return true;
}

template<class real>
bool allclose(const sqcu::DeviceVectorType<real> &dvec, const sqaod::VectorType<real> &hvec, real epsiron);


#endif
