#include "DeviceMath.h"
#include "Assertion.h"
#include <algorithm>

using namespace sqaod_cuda;
using sq::Dim;
using sq::SizeType;

template<class real>
void DeviceMathType<real>::broadcastToDiagonal(DeviceMatrix *A, real v) {
    assertValidMatrix(*A, __func__);
    devCopy_.broadcastToDiagonal(A, v, 0);
}

template<class real>
void DeviceMathType<real>::scale(DeviceScalar *y, real alpha, const DeviceScalar &x,
                                 real addAssignFactor) {
    devAlloc_->allocateIfNull(y);
    devKernels_.scale(y, alpha, x, addAssignFactor);
}

template<class real>
void DeviceMathType<real>::scale(DeviceVector *y, real alpha, const DeviceVector &x,
                                 real addAssignFactor) {
    devAlloc_->allocateIfNull(y, x.size);
    assertSameShape(*y, x, __func__);
    devKernels_.scale(y, alpha, x, addAssignFactor);
}

template<class real>
void DeviceMathType<real>::scale(DeviceMatrix *B, real alpha, const DeviceMatrix &A) {
    devAlloc_->allocateIfNull(B, A.dim());
    assertSameShape(*B, A, __func__);
    devKernels_.scale(B, alpha, A, real(0.));
}

template<class real>
void DeviceMathType<real>::scaleBroadcast(DeviceVector *y, real alpha, const DeviceScalar &x,
                                          real addAssignFactor) {
    assertValidScalar(x, __func__);
    assertValidVector(*y, __func__);
    devKernels_.scaleBroadcast(y, alpha, x, addAssignFactor);
}

template<class real>
void DeviceMathType<real>::scaleBroadcast(DeviceMatrix *A, real alpha, const DeviceVector &x,
                                          BatchOp op, real addAssignFactor) {
    assertValidMatrix(*A, __func__);
    if (op == opRowwise) {
        abortIf(A->cols != x.size, "Cols of matrix does not match vector length.");
        devKernels_.scaleBroadcastToRows(A, alpha, x, addAssignFactor);
    }
    else if (op == opColwise) {
        throwErrorIf(A->rows != x.size, "Rows of matrix does not match vector length.");
        devKernels_.scaleBroadcastToColumns(A, alpha, x, addAssignFactor);
    }
    else {
        abort_("Unknown matrix op.");
    }
}
    
template<class real>
void DeviceMathType<real>::sum(DeviceScalar *s, real alpha, const DeviceVector &x,
                               real addAssignFactor) {
    devAlloc_->allocateIfNull(s);
    devKernels_.sum(s, alpha, x, addAssignFactor);
}

template<class real>
void DeviceMathType<real>::sum(DeviceScalar *s, real alpha, const DeviceMatrix &dmat,
                               real addAssignFactor) {
    devAlloc_->allocateIfNull(s);
    devKernels_.sum(s, alpha, dmat, addAssignFactor);
}

template<class real>
void DeviceMathType<real>::sumDiagonal(DeviceScalar *s, const DeviceMatrix &dmat) {
    devAlloc_->allocateIfNull(s);
    devKernels_.sumDiagonal(s, 1., dmat, 0, 0.);
}

template<class real>
void DeviceMathType<real>::sumBatched(DeviceVector *vec,
                                      real alpha, const DeviceMatrix &A, BatchOp op) {
    const DeviceMatrix *dmat;
    if (op == opColwise) {
        Dim trDim = A.dim().transpose();
        DeviceMatrix *transposed = tempDeviceMatrix(trDim);
        transpose(transposed, A);
        dmat = transposed;
    }
    else if (op == opRowwise) {
        dmat = &A;
    }
    else {
        dmat = NULL; /* to supress warning with g++ */
        abort_("Invalid BatchOp.");
    }
    devAlloc_->allocateIfNull(vec, dmat->rows);
    assertValidVector(*vec, dmat->rows, __func__);
    devKernels_.sumRowwise(vec, alpha, *dmat);
}

template<class real>
void DeviceMathType<real>::dot(DeviceScalar *z,
                               real alpha, const DeviceVector &x, const DeviceVector &y,
                               real addAssignFactor) {
    devAlloc_->allocateIfNull(z);
    assertSameShape(x, y, __func__);
    devKernels_.dot(z, alpha, x, y, addAssignFactor);
}

template<class real>
void DeviceMathType<real>::dotBatched(DeviceVector *z,
                                      real alpha, const DeviceMatrix &A, MatrixOp opA,
                                      const DeviceMatrix &B, MatrixOp opB) {
    const DeviceMatrix *dMat0, *dMat1;
    if (opA == opTranspose) {
        DeviceMatrix *dAt;
        dAt = tempDeviceMatrix(A.dim().transpose());
        transpose(dAt, A);
        dMat0 = dAt;
    }
    else {
        assert(opA == opNone);
        dMat0 = &A;
    }
    if (opB == opTranspose) {
        DeviceMatrix *dBt;
        dBt = tempDeviceMatrix(B.dim().transpose());
        transpose(dBt, A);
        dMat1 = dBt;
    }
    else {
        assert(opB == opNone);
        dMat1 = &B;
    }
    assertSameShape(*dMat0, *dMat1, __func__);
    devAlloc_->allocateIfNull(z, dMat0->rows);
    assertValidVector(*z, dMat0->rows, __func__);
    devKernels_.dotRowwise(z, alpha, *dMat0, *dMat1);
}

template<class real>
void DeviceMathType<real>::mvProduct(DeviceVector *y, real alpha,
                                     const DeviceMatrix &A, MatrixOp opA, const DeviceVector &x) {
    SizeType size = getProductShape(A, opA, x);
    devAlloc_->allocateIfNull(y, size);
    assertValidVector(*y, size, __func__);
    const DeviceScalar &d_alpha = d_const(alpha);
    gemv(opA, d_alpha, A, x, d_zero(), *y);
}

template<class real>
void DeviceMathType<real>::vmProduct(DeviceVector *y, real alpha,
                                     const DeviceVector &x, const DeviceMatrix &A, MatrixOp opA,
                                     real addAssignFactor) {
    const DeviceScalar &d_alpha = d_const(alpha);
    const DeviceScalar &d_factor = d_const(addAssignFactor);
    opA = (opA == opNone) ? opTranspose : opNone;
    SizeType size = getProductShape(A, opA, x);
    devAlloc_->allocateIfNull(y, size);
    assertValidVector(*y, size, __func__);
    gemv(opA, d_alpha, A, x, d_factor, *y);
}

template<class real>
void DeviceMathType<real>::mmProduct(DeviceMatrix *C, real alpha,
                                     const DeviceMatrix &A, MatrixOp opA,
                                     const DeviceMatrix &B, MatrixOp opB) {
    Dim dim = getProductShape(A, opA, B, opB);
    abortIf(dim == Dim(), "shape mismatch on matrix-matrix multiplication.");
    devAlloc_->allocateIfNull(C, dim);
    assertValidMatrix(*C, dim, __func__);

    const DeviceScalar &d_alpha = d_const(alpha);
    gemm(opA, opB, d_alpha, A, B, d_zero(), *C);
}
    
template<class real>
void DeviceMathType<real>::vmvProduct(DeviceScalar *z, real alpha,
                                      const DeviceVector &y, const DeviceMatrix &A,
                                      const DeviceVector &x) {
    DeviceVector *Ax = tempDeviceVector(A.rows);
    gemv(opNone, d_one(), A, x, d_zero(), *Ax);

    devAlloc_->allocateIfNull(z);
    dot(z, alpha, y, *Ax);
}

template<class real>
void DeviceMathType<real>::vmvProductBatched(DeviceVector *z, real alpha,
                                             const DeviceMatrix &y,
                                             const DeviceMatrix &A,
                                             const DeviceMatrix &x) {
    abortIf(x.rows != y.rows, "shape mismatch on batched VxMxV product.");
    abortIf((y.cols != A.rows) || (A.cols != x.cols), "shape mismatch on batched VxMxV product.");

    Dim dim = getProductShape(x, opNone, A, opTranspose);
    abortIf(dim == Dim(), "shape mismatch on batched VxMxV product.");
        
    DeviceMatrix *xA = tempDeviceMatrix(dim);
    abortIf(xA->cols != y.cols, "shape mismatch on batched VxMxV product.");
    devAlloc_->allocateIfNull(z, x.rows);
    assertValidVector(*z, x.rows, __func__);
    
    gemm(opNone, opTranspose, d_one(), x, A, d_zero(), *xA);
    dotBatched(z, alpha, *xA, opNone, y, opNone);
}

template<class real>
void DeviceMathType<real>::mmmProduct(DeviceMatrix *z, real alpha,
                                      const DeviceMatrix &y, MatrixOp opy,
                                      const DeviceMatrix &A, MatrixOp opA,
                                      const DeviceMatrix &x, MatrixOp opx) {
    Dim dimAx = getProductShape(A, opA, x, opx);
    DeviceMatrix *Ax =  tempDeviceMatrix(dimAx);
    mmProduct(Ax, 1., A, opA, x, opx);
    mmProduct(z, 1., y, opy, *Ax, opNone);
}

template<class real>
void DeviceMathType<real>::min(DeviceScalar *s, const DeviceMatrix &A) {
    devAlloc_->allocateIfNull(s);
    devKernels_.min(s, A);
}

template<class real>
void DeviceMathType<real>::min(DeviceScalar *s, const DeviceVector &x) {
    devAlloc_->allocateIfNull(s);
    devKernels_.min(s, x);
}

template<class real>
void DeviceMathType<real>::transpose(DeviceMatrix *dAt, const DeviceMatrix &A) {
    Dim dim = getMatrixShape(A, opTranspose);
    devAlloc_->allocateIfNull(dAt, dim);
    assertValidMatrix(*dAt, dim, __func__);
    devKernels_.transpose(dAt, A);
}

template<class real>
void DeviceMathType<real>::symmetrize(DeviceMatrix *dAsym, const DeviceMatrix &A) {
    Dim dim = getMatrixShape(A, opTranspose);
    devAlloc_->allocateIfNull(dAsym, dim);
    assertValidMatrix(*dAsym, dim, __func__);
    devKernels_.symmetrize(dAsym, A);
}


/* Matrix shape */
template<class real>
sq::Dim DeviceMathType<real>::getMatrixShape(const DeviceMatrix &A, MatrixOp opA) {
    Dim dim;
    dim.rows = opA == opNone ? A.rows : A.cols;
    dim.cols = opA == opNone ? A.cols : A.rows;
    return dim;
}


template<class real>
sq::Dim DeviceMathType<real>::getProductShape(const DeviceMatrix &A, MatrixOp opA,
                                                 const DeviceMatrix &B, MatrixOp opB) {
    Dim Adim = getMatrixShape(A, opA);
    Dim Bdim = getMatrixShape(B, opB);
    if (Adim.cols != Bdim.rows)
        return Dim();
    return Dim(Adim.rows, Bdim.cols);
}

template<class real>
SizeType DeviceMathType<real>::getProductShape(const DeviceMatrix &A, MatrixOp opA,
                                               const DeviceVector &x) {
    Dim Adim = getMatrixShape(A, opA);
    throwErrorIf(Adim.cols != x.size, "Shape does not match on matrix-vector multiplication.");  
    return Adim.rows;
}

template<class real>
SizeType DeviceMathType<real>::getProductShape(const DeviceVector &x,
                                               const DeviceMatrix &A, MatrixOp opA) {
    Dim Adim = getMatrixShape(A, opA);
    throwErrorIf(Adim.rows != x.size, "Shape does not match on vector-matrix multiplication.");  
    return Adim.cols;
}

template<class real>
void DeviceMathType<real>::gemv(MatrixOp op, const DeviceScalar &d_alpha,
                                const DeviceMatrix &A, const DeviceVector &x,
                                const DeviceScalar &d_beta, DeviceVector &y) {
    /* transpose since A is in row-major format, though cublas accepts column-major format. */
    cublasOperation_t cop = (op == opNone) ? CUBLAS_OP_T : CUBLAS_OP_N;
    devKernels_.gemv(cop, A.cols, A.rows,
                     d_alpha.d_data, A.d_data, A.stride,
                     x.d_data, d_beta.d_data, y.d_data);
}

template<class real>
void DeviceMathType<real>::gemm(MatrixOp opA, MatrixOp opB,
                                const DeviceScalar &d_alpha, const DeviceMatrix &A, const DeviceMatrix &B,
                                const DeviceScalar &d_beta, DeviceMatrix &C) {
    /* To get C in row-major format, actual calculation is CT = BT x AT.
     * We need to transpose to fix column-major format, and transpose again to get C in row-major format, thus, copA, copA is not transposed. */
    cublasOperation_t copA = (opA == opNone) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t copB = (opB == opNone) ? CUBLAS_OP_N : CUBLAS_OP_T;
    Dim Adim = getMatrixShape(A, opA);
    Dim Bdim = getMatrixShape(B, opB);
    abortIf(Adim.cols != Bdim.rows, "shape mismatch on matrix-matrix multiplication");

    /* leading diimension */
    int ldb = B.stride;
    int lda = A.stride;
    int ldc = C.stride;

    devKernels_.gemm(copB, copA, Bdim.cols, Adim.rows, Adim.cols,
                     d_alpha.d_data, B.d_data, ldb, A.d_data, lda,
                     d_beta.d_data, C.d_data, ldc);
}

template<class real>
DeviceMathType<real>::DeviceMathType() {
    devStream_ = NULL;
}

template<class real>
DeviceMathType<real>::DeviceMathType(Device &device, DeviceStream *devStream){
    devStream_ = NULL;
    assignDevice(device, devStream);
}

template<class real>
void DeviceMathType<real>::assignDevice(Device &device, DeviceStream *devStream) {
    throwErrorIf(devStream_ != NULL, "Device already assigned.");
    if (devStream == NULL)
        devStream = device.defaultStream();
    devStream_ = devStream;

    devAlloc_ = device.objectAllocator();
    devConst_ = device.constScalars<real>();
    devCopy_.assignDevice(device, devStream);
    devKernels_.assignStream(devStream);
}


template struct sqaod_cuda::DeviceMathType<float>;
template struct sqaod_cuda::DeviceMathType<double>;

