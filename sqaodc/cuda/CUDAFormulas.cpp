#include "CUDAFormulas.h"
#include <sqaodc/common/ShapeChecker.h>


namespace sqint = sqaod_internal;
namespace sqcu = sqaod_cuda;;

template<class real> void sqcu::CUDADenseGraphFormulas<real>::
calculate_E(real *E, const HostMatrix &W, const HostVector &x) {
    sqint::validateScalar(E, __func__);
    DeviceScalar *d_E = devStream->tempDeviceScalar<real>();
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceVector *d_x = devStream->tempDeviceVector<real>(x.size);
    devCopy(d_W, W);
    devCopy(d_x, x);
    formulas.calculate_E(d_E, *d_W, *d_x);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real> void sqcu::CUDADenseGraphFormulas<real>::
calculate_E(HostVector *E, const HostMatrix &W, const HostMatrix &x) {
    sqint::quboShapeCheck(W, x, __func__);
    sqint::validateScalar(E, __func__);

    DeviceVector *d_E = devStream->tempDeviceVector<real>(x.rows);
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceMatrix *d_x = devStream->tempDeviceMatrix<real>(x.dim());
    devCopy(d_W, W);
    devCopy(d_x, x);
    formulas.calculate_E(d_E, *d_W, *d_x);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real> void sqcu::CUDADenseGraphFormulas<real>::
calculateHamiltonian(HostVector *h, HostMatrix *J, real *c, const HostMatrix &W) {
    sqint::quboShapeCheck(W, __func__);
    sqint::prepVector(h, W.rows, __func__);
    sqint::prepMatrix(J, W.dim(), __func__);
    sqint::validateScalar(c, __func__);
    
    DeviceVector *d_h = devStream->tempDeviceVector<real>(W.rows);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    devCopy(d_W, W);
    formulas.calculateHamiltonian(d_h, d_J, d_c, *d_W);
    devCopy(h, *d_h);
    devCopy(J, *d_J);
    devCopy(c, *d_c);
    devStream->synchronize();
}

template<class real> void sqcu::CUDADenseGraphFormulas<real>::
calculate_E(real *E,
            const HostVector &h, const HostMatrix &J, real c,
            const HostVector &q) {
    sqint::isingModelShapeCheck(h, J, c, q, __func__);
    sqint::validateScalar(E, __func__);
    
    DeviceScalar *d_E = devStream->tempDeviceScalar<real>();
    DeviceVector *d_h = devStream->tempDeviceVector<real>(h.size);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(J.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceVector *d_q = devStream->tempDeviceVector<real>(q.size);
    devCopy(d_h, h);
    devCopy(d_J, J);
    devCopy(d_c, c);
    devCopy(d_q, q);
    formulas.calculate_E(d_E, *d_h, *d_J, *d_c, *d_q);
    devCopy(E, *d_E);
    devStream->synchronize();
}

template<class real> void sqcu::CUDADenseGraphFormulas<real>::
calculate_E(HostVector *E,
            const HostVector &h, const HostMatrix &J, real c,
            const HostMatrix &q) {
    sqint::isingModelShapeCheck(h, J, c, q,  __func__);
    sqint::prepVector(E, q.rows, __func__);

    DeviceVector *d_E = devStream->tempDeviceVector<real>(q.rows);
    DeviceVector *d_h = devStream->tempDeviceVector<real>(h.size);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(J.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceMatrix *d_q = devStream->tempDeviceMatrix<real>(q.dim());
    devCopy(d_h, h);
    devCopy(d_J, J);
    devCopy(d_c, c);
    devCopy(d_q, q);
    formulas.calculate_E(d_E, *d_h, *d_J, *d_c, *d_q);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real>
sqcu::CUDADenseGraphFormulas<real>::CUDADenseGraphFormulas() {
    devStream = NULL;
}


template<class real>
void sqcu::CUDADenseGraphFormulas<real>::assignDevice(sqaod::cuda::Device &device) {
    throwErrorIf(devStream != NULL, "Device already assigned.");
    sqcu::Device &dev = static_cast<sqcu::Device&>(device);
    devStream = dev.defaultStream();
    devCopy.assignDevice(dev, devStream);
    formulas.assignDevice(dev, devStream);
}


/* Bipartite graph */

template<class real> void sqcu::CUDABipartiteGraphFormulas<real>::
calculate_E(real *E,
            const HostVector &b0, const HostVector &b1, const HostMatrix &W,
            const HostVector &x0, const HostVector &x1) {
    sqint::quboShapeCheck(b0, b1, W, x0, x1, __func__);
    sqint::validateScalar(E, __func__);

    DeviceScalar *d_E = devStream->tempDeviceScalar<real>();
    DeviceVector *d_b0 = devStream->tempDeviceVector<real>(b0.size);
    DeviceVector *d_b1 = devStream->tempDeviceVector<real>(b1.size);
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceVector *d_x0 = devStream->tempDeviceVector<real>(x0.size);
    DeviceVector *d_x1 = devStream->tempDeviceVector<real>(x1.size);
    devCopy(d_b0, b0);
    devCopy(d_b1, b1);
    devCopy(d_W, W);
    devCopy(d_x0, x0);
    devCopy(d_x1, x1);
    formulas.calculate_E(d_E, *d_b0, *d_b1, *d_W, *d_x0, *d_x1);
    devCopy(E, *d_E);
    devStream->synchronize();
}

template<class real> void sqcu::CUDABipartiteGraphFormulas<real>::
calculate_E(HostVector *E,
            const HostVector &b0, const HostVector &b1, const HostMatrix &W,
            const HostMatrix &x0, const HostMatrix &x1) {
    sqint::quboShapeCheck(b0, b1, W, x0, x1, __func__);
    sqint::prepVector(E, x1.rows, __func__);

    DeviceVector *d_E = devStream->tempDeviceVector<real>(x0.rows);
    DeviceVector *d_b0 = devStream->tempDeviceVector<real>(b0.size);
    DeviceVector *d_b1 = devStream->tempDeviceVector<real>(b1.size);
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceMatrix *d_x0 = devStream->tempDeviceMatrix<real>(x0.dim());
    DeviceMatrix *d_x1 = devStream->tempDeviceMatrix<real>(x1.dim());
    devCopy(d_b0, b0);
    devCopy(d_b1, b1);
    devCopy(d_W, W);
    devCopy(d_x0, x0);
    devCopy(d_x1, x1);
    formulas.calculate_E(d_E, *d_b0, *d_b1, *d_W, *d_x0, *d_x1);
    devCopy(E, *d_E);
    devStream->synchronize();
}

template<class real>
void sqcu::CUDABipartiteGraphFormulas<real>::
calculate_E_2d(HostMatrix *E,
               const HostVector &b0, const HostVector &b1, const HostMatrix &W,
               const HostMatrix &x0, const HostMatrix &x1) {
    sqint::quboShapeCheck_2d(b0, b1, W, x0, x1, __func__);
    sqint::prepMatrix(E, sq::Dim(x1.rows, x0.rows), __func__);

    DeviceMatrix *d_E = devStream->tempDeviceMatrix<real>(x1.rows, x0.rows);
    DeviceVector *d_b0 = devStream->tempDeviceVector<real>(b0.size);
    DeviceVector *d_b1 = devStream->tempDeviceVector<real>(b1.size);
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceMatrix *d_x0 = devStream->tempDeviceMatrix<real>(x0.dim());
    DeviceMatrix *d_x1 = devStream->tempDeviceMatrix<real>(x1.dim());
    devCopy(d_b0, b0);
    devCopy(d_b1, b1);
    devCopy(d_W, W);
    devCopy(d_x0, x0);
    devCopy(d_x1, x1);
    formulas.calculate_E_2d(d_E, *d_b0, *d_b1, *d_W, *d_x0, *d_x1);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real> void sqcu::CUDABipartiteGraphFormulas<real>::
calculateHamiltonian(HostVector *h0, HostVector *h1, HostMatrix *J, real *c,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W) {
    sqint::quboShapeCheck(b0, b1, W, __func__);
    sqint::prepVector(h0, b0.size, __func__);
    sqint::prepVector(h1, b1.size, __func__);
    sqint::prepMatrix(J, W.dim(), __func__);
    sqint::validateScalar(c, __func__);
    
    DeviceVector *d_h0 = devStream->tempDeviceVector<real>(b0.size);
    DeviceVector *d_h1 = devStream->tempDeviceVector<real>(b1.size);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceVector *d_b0 = devStream->tempDeviceVector<real>(b0.size);
    DeviceVector *d_b1 = devStream->tempDeviceVector<real>(b1.size);
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());

    devCopy(d_b0, b0);
    devCopy(d_b1, b1);
    devCopy(d_W, W);
    formulas.calculateHamiltonian(d_h0, d_h1, d_J, d_c, *d_b0, *d_b1, *d_W);
    devCopy(h0, *d_h0);
    devCopy(h1, *d_h1);
    devCopy(J, *d_J);
    devCopy(c, *d_c);
    devStream->synchronize();
}


template<class real>
void sqcu::CUDABipartiteGraphFormulas<real>::
calculate_E(real *E,
            const HostVector &h0, const HostVector &h1, const HostMatrix &J, real c,
            const HostVector &q0, const HostVector &q1) {
    sqint::isingModelShapeCheck(h0, h1, J, c, q0, q1, __func__);
    sqint::validateScalar(E, __func__);

    DeviceScalar *d_E = devStream->tempDeviceScalar<real>();
    DeviceVector *d_h0 = devStream->tempDeviceVector<real>(h0.size);
    DeviceVector *d_h1 = devStream->tempDeviceVector<real>(h1.size);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(J.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceVector *d_q0 = devStream->tempDeviceVector<real>(q0.size);
    DeviceVector *d_q1 = devStream->tempDeviceVector<real>(q1.size);

    devCopy(d_h0, h0);
    devCopy(d_h1, h1);
    devCopy(d_J, J);
    devCopy(d_c, c);
    devCopy(d_q0, q0);
    devCopy(d_q1, q1);
    formulas.calculate_E(d_E, *d_h0, *d_h1, *d_J, *d_c, *d_q0, *d_q1);
    devCopy(E, *d_E);
    devStream->synchronize();
}

template<class real> void sqcu::CUDABipartiteGraphFormulas<real>::
calculate_E(HostVector *E,
            const HostVector &h0, const HostVector &h1, const HostMatrix &J, real c,
            const HostMatrix &q0, const HostMatrix &q1) {
    sqint::isingModelShapeCheck(h0, h1, J, c, q0, q1, __func__);
    sqint::prepVector(E, q0.rows, __func__);

    DeviceVector *d_E = devStream->tempDeviceVector<real>(q0.rows);
    DeviceVector *d_h0 = devStream->tempDeviceVector<real>(h0.size);
    DeviceVector *d_h1 = devStream->tempDeviceVector<real>(h1.size);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(J.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceMatrix *d_q0 = devStream->tempDeviceMatrix<real>(q0.dim());
    DeviceMatrix *d_q1 = devStream->tempDeviceMatrix<real>(q1.dim());

    devCopy(d_h0, h0);
    devCopy(d_h1, h1);
    devCopy(d_J, J);
    devCopy(d_c, c);
    devCopy(d_q0, q0);
    devCopy(d_q1, q1);
    formulas.calculate_E(d_E, *d_h0, *d_h1, *d_J, *d_c, *d_q0, *d_q1);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real>
sqcu::CUDABipartiteGraphFormulas<real>::CUDABipartiteGraphFormulas() {
    devStream = NULL;
}

template<class real>
void sqcu::CUDABipartiteGraphFormulas<real>::assignDevice(sqaod::cuda::Device &device) {
    throwErrorIf(devStream != NULL, "Device already assigned.");
    sqcu::Device &dev = static_cast<sqcu::Device&>(device);
    devStream = dev.defaultStream();
    devCopy.assignDevice(dev, devStream);
    formulas.assignDevice(dev, devStream);
}


template struct sqcu::CUDADenseGraphFormulas<double>;
template struct sqcu::CUDADenseGraphFormulas<float>;
template struct sqcu::CUDABipartiteGraphFormulas<double>;
template struct sqcu::CUDABipartiteGraphFormulas<float>;
