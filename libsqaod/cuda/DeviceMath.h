#ifndef CUDA_DEVICEMATH_H__
#define CUDA_DEVICEMATH_H__

#include <cuda/DeviceMatrix.h>
#include <cuda/DeviceStream.h>
#include <cuda/DeviceCopy.h>

namespace sqaod_cuda {

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
    typedef unsigned int uint;
    
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;

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
                    const DeviceMatrix &B, MatrixOp opB, real addAssignFactor = 0.);
    
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


    void initialize(DeviceStream *devStream);
    void uninitialize();

    /* get matrix shape resulting from matrix arithmetic */
    sqaod::Dim getProductShape(const DeviceMatrix &A, MatrixOp opA,
                               const DeviceMatrix &B, MatrixOp opB);
    sqaod::Dim getProductShape(const DeviceMatrix &A, MatrixOp opA, const DeviceVector &x);
    sqaod::Dim getProductShape(const DeviceVector &x, const DeviceMatrix &A, MatrixOp opA);

    /* Device Const */
    const DeviceScalar &deviceConst(real c);
    const DeviceScalar &d_one();
    const DeviceScalar &d_zero();

    /* temporary objects */
    DeviceMatrix *tempDeviceMatrix(int rows, int cols, const char *signature = NULL);
    DeviceMatrix *tempDeviceMatrix(const sqaod::Dim &dim, const char *signature = NULL);
    DeviceVector *tempDeviceVector(uint size, const char *signature = NULL);
    DeviceScalar *tempDeviceScalar(int rows, int cols, const char *signature = NULL);
    void *tempAllocate(uint size);

    /* CUDA funcs */
    void min(DeviceScalar *s, const DeviceMatrix &A);
    void transpose(DeviceMatrix *dAt, const DeviceMatrix &A);

    void scale(real *d_y, real alpha, const real *d_x, uint size);
    void scaleBroadcast(real *d_x, real alpha, const real *d_c, uint size, real addAssignFactor);
    void scaleBroadcastVector(real *d_A, real alpha, const real *d_x, uint size,
                              uint nBatch, real addAssignFactor);
    void scaleBroadcastScalars(real *d_A, real alpha, const real *d_x, uint size,
                               uint nBatch, real addAssignFactor);

    void sum(real *d_dst, real alpha, const real *d_x, uint size, real addAssignFactor);
    void sumGather(real *d_dst, real alpha, const real *d_x, uint size, uint stride, int offset);
    void sumBatched(real *d_x, real alpha, const real *d_A, uint size, uint nBatch);
    void dot(real *d_c, real alpha, const real *d_x, const real *d_y, uint size,
             real addAssignFactor);
    void dotBatched(real *d_z, real alpha, const real *d_x, const real *d_y, uint size,
                    uint nBatch, real addAssignFactor);

    void gemv(MatrixOp op, const DeviceScalar &d_alpha,
              const DeviceMatrix &A, const DeviceVector &x,
              const DeviceScalar &d_beta, DeviceVector &y);
    void gemm(MatrixOp opA, MatrixOp opB,
              const DeviceScalar &d_alpha, const DeviceMatrix &A, const DeviceMatrix &B,
              const DeviceScalar &d_beta, DeviceMatrix &C);
    
private:
    DeviceCopy devCopy_;
    DeviceStream *devStream_;
    Device *device_;
};



}  // namespace sqaod_cuda

#endif
