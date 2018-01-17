#ifndef CUDA_DEVICEMATH_H__
#define CUDA_DEVICEMATH_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceObjectAllocator.h>
#include <cuda/DeviceStream.h>
#include <cuda/DeviceCopy.h>
#include <cuda/DeviceMathKernels.h>

namespace sqaod_cuda {

class Device;

enum MatrixOp {
    opNone,
    opTranspose,
};

enum VectorOp {
    opRowVector,
    opColumnVector,
};

enum BatchOp {
    opRowwise,
    opColwise,
};


template<class real>
struct DeviceMathType {
    typedef sqaod::SizeType SizeType;
    
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceObjectAllocatorType<real> DeviceObjectAllocator;
    typedef DeviceMathKernelsType<real> DeviceMathKernels;

    void setToDiagonals(DeviceMatrix *V, real v);
    
    void scale(DeviceScalar *y, real alpha, const DeviceScalar &x, real addAssignFactor = 0.);
    void scale(DeviceVector *y, real alpha, const DeviceVector &x, real addAssignFactor = 0.);
    void scale(DeviceMatrix *B, real alpha, const DeviceMatrix &A);
    void scaleBroadcast(DeviceVector *y, real alpha, const DeviceScalar &x,
                        real addAssignFactor = 0.);
    void scaleBroadcast(DeviceMatrix *y, real alpha, const DeviceVector &x, BatchOp op,
                        real addAssignFactor = 0.);
    
    void sum(DeviceScalar *s, real alpha, const DeviceVector &x, real addAssignFactor = 0.);
    void sum(DeviceScalar *s, real alpha, const DeviceMatrix &dmat, real addAssignFactor = 0.);
    void sumDiagonals(DeviceScalar *s, const DeviceMatrix &dmat);

    void sumBatched(DeviceVector *vec, real alpha, const DeviceMatrix &dmat, BatchOp op);
    
    void dot(DeviceScalar *z, real alpha, const DeviceVector &x, const DeviceVector &y,
             real addAssignFactor = 0.);

    /* dot(row, row) */
    void dotBatched(DeviceVector *z,
                    real alpha, const DeviceMatrix &A, MatrixOp opA,
                    const DeviceMatrix &B, MatrixOp opB);
    
    void mvProduct(DeviceVector *y,
                   real alpha, const DeviceMatrix &A, MatrixOp opA, const DeviceVector &x);
    void vmProduct(DeviceVector *y,
                   real alpha, const DeviceVector &x, const DeviceMatrix &A, MatrixOp opA,
                   real addAssignFactor = 0.);
    void mmProduct(DeviceMatrix *C, real alpha,
                   const DeviceMatrix &A, MatrixOp opA, const DeviceMatrix &B, MatrixOp opB);
    
    void vmvProduct(DeviceScalar *z,
                    real alpha, const DeviceVector &y,
                    const DeviceMatrix &A, const DeviceVector &x);

    void batchedVmvProduct(DeviceVector *z, real alpha,
                           const DeviceMatrix &y, const DeviceMatrix &A, const DeviceMatrix &x);

    void mmmProduct(DeviceMatrix *z, real alpha,
                    const DeviceMatrix &y, MatrixOp opy,
                    const DeviceMatrix &A, MatrixOp opA,
                    const DeviceMatrix &x, MatrixOp opx);

    void min(DeviceScalar *s, const DeviceMatrix &A);
    void transpose(DeviceMatrix *dAt, const DeviceMatrix &A);
    
    /* get matrix shape resulting from matrix arithmetic */
    sqaod::Dim getMatrixShape(const DeviceMatrix &A, MatrixOp opA);
    sqaod::Dim getProductShape(const DeviceMatrix &A, MatrixOp opA,
                               const DeviceMatrix &B, MatrixOp opB);
    SizeType getProductShape(const DeviceMatrix &A, MatrixOp opA, const DeviceVector &x);
    SizeType getProductShape(const DeviceVector &x, const DeviceMatrix &A, MatrixOp opA);

    /* Device Const */
    const DeviceScalar &d_const(real c);
    const DeviceScalar &d_one();
    const DeviceScalar &d_zero();

    /* temporary objects */
    DeviceMatrix *tempDeviceMatrix(int rows, int cols, const char *signature = NULL);
    DeviceMatrix *tempDeviceMatrix(const sqaod::Dim &dim, const char *signature = NULL);
    DeviceVector *tempDeviceVector(SizeType size, const char *signature = NULL);
    DeviceScalar *tempDeviceScalar(const char *signature = NULL);
    void *tempAllocate(SizeType size);
    /* objects on heap */
    DeviceMatrix *newDeviceMatrix(int rows, int cols, const char *signature = NULL);
    DeviceMatrix *newDeviceMatrix(const sqaod::Dim &dim, const char *signature = NULL);
    DeviceVector *newDeviceVector(SizeType size, const char *signature = NULL);
    DeviceScalar *newDeviceScalar(const char *signature = NULL);
    void *allocate(size_t size);

    /* BLAS */
    void gemv(MatrixOp op, const DeviceScalar &d_alpha,
              const DeviceMatrix &A, const DeviceVector &x,
              const DeviceScalar &d_beta, DeviceVector &y);
    void gemm(MatrixOp opA, MatrixOp opB,
              const DeviceScalar &d_alpha, const DeviceMatrix &A, const DeviceMatrix &B,
              const DeviceScalar &d_beta, DeviceMatrix &C);


    DeviceMathType();
    DeviceMathType(Device &device, DeviceStream *devStream = NULL);

    void assignDevice(Device &device, DeviceStream *devStream = NULL);
    
private:
    DeviceMathKernels devKernels_;
    DeviceCopyType<real> devCopy_;
    DeviceObjectAllocator *devAlloc_;
    DeviceStream *devStream_;
};



/* inline method implimentations */

/* Device Const */
template<class real> inline
const DeviceScalarType<real> &DeviceMathType<real>::d_const(real c) {
    return devAlloc_->d_const(c);
}

template<class real> inline
const DeviceScalarType<real> &DeviceMathType<real>::d_one() {
    return devAlloc_->d_one();
}

template<class real> inline
const DeviceScalarType<real> &DeviceMathType<real>::d_zero() {
    return devAlloc_->d_zero();
}

/* temporary objects */
template<class real> inline
DeviceMatrixType<real> *DeviceMathType<real>::tempDeviceMatrix(int rows, int cols,
                                                               const char *signature) {
    return devStream_->tempDeviceMatrix<real>(rows, cols, signature);
}

template<class real> inline
DeviceMatrixType<real> *DeviceMathType<real>::tempDeviceMatrix(const sqaod::Dim &dim,
                                                               const char *signature) {
    return devStream_->tempDeviceMatrix<real>(dim.rows, dim.cols, signature);
}

template<class real> inline
DeviceVectorType<real> *DeviceMathType<real>::tempDeviceVector(SizeType size,
                                                               const char *signature) {
    return devStream_->tempDeviceVector<real>(size, signature);
}

template<class real> inline
DeviceScalarType<real> *DeviceMathType<real>::tempDeviceScalar(const char *signature) {
    return devStream_->tempDeviceScalar<real>(signature);
}

template<class real> inline
void *DeviceMathType<real>::tempAllocate(SizeType size) {
    return devStream_->allocate(size);
}


}  // namespace sqaod_cuda

#endif
