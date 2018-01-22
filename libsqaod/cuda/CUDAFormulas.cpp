#include "CUDAFormulas.h"
#include <iostream>
#include <float.h>


using namespace sqaod_cuda;


template<class real>
void CUDADGFuncs<real>::calculate_E(DeviceScalar *E,
                                    const DeviceMatrix &W, const DeviceVector &x) {
    devMath.vmvProduct(E, 1., x, W, x);
}


template<class real>
void CUDADGFuncs<real>::calculate_E(DeviceVector *E,
                                    const DeviceMatrix &W, const DeviceMatrix &x) {
    devMath.vmvProductBatched(E, 1., x, W, x);
}


template<class real>
void CUDADGFuncs<real>::calculate_hJc(DeviceVector *h, DeviceMatrix *J, DeviceScalar *c,
                                      const DeviceMatrix &W) {
    devMath.sumBatched(h, 0.5, W, opColwise);
    devMath.scale(J, 0.25, W);

    devMath.sumDiagonals(c, *J);
    devMath.sum(c, 1., *J, 1.);
    devMath.setToDiagonals(J, 0.);
}

template<class real>
void CUDADGFuncs<real>::calculate_E(DeviceScalar *E,
                                    const DeviceVector &h, const DeviceMatrix &J,
                                    const DeviceScalar &c, const DeviceVector &q) {
    devMath.vmvProduct(E, 1., q, J, q);
    devMath.dot(E, 1., h, q, 1.);
    devMath.scale(E, 1., c, 1.);
}

template<class real>
void CUDADGFuncs<real>::calculate_E(DeviceVector *E,
                                    const DeviceVector &h, const DeviceMatrix &J,
                                    const DeviceScalar &c, const DeviceMatrix &q) {
    devMath.vmvProductBatched(E, 1., q, J, q);
    devMath.vmProduct(E, 1., h, q, opTranspose, 1.);
    devMath.scaleBroadcast(E, 1., c, 1.);
}

template<class real>
void CUDADGFuncs<real>::assignDevice(Device &device, DeviceStream *stream) {
    devMath.assignDevice(device, stream);
}



/* rbm */

template<class real>
void CUDABGFuncs<real>::calculate_E(DeviceScalar *E,
                                      const DeviceVector &b0, const DeviceVector &b1,
                                      const DeviceMatrix &W,
                                      const DeviceVector &x0, const DeviceVector &x1) {
    devMath.vmvProduct(E, 1., x1, W, x0);
    devMath.dot(E, 1., b0, x0, 1.);
    devMath.dot(E, 1., b1, x1, 1.);
}

template<class real>
void CUDABGFuncs<real>::calculate_E(DeviceVector *E,
                                    const DeviceVector &b0, const DeviceVector &b1,
                                    const DeviceMatrix &W,
                                    const DeviceMatrix &x0, const DeviceMatrix &x1) {

    devMath.vmvProductBatched(E, 1., x1, W, x0);
    DeviceVector *bx0 = devMath.tempDeviceVector(x0.rows, __func__);
    DeviceVector *bx1 = devMath.tempDeviceVector(x1.rows, __func__);
    devMath.vmProduct(bx0, 1., b0, x0, opTranspose);
    devMath.vmProduct(bx1, 1., b1, x1, opTranspose);
    devMath.scale(E, 1., *bx0, 1.);
    devMath.scale(E, 1., *bx1, 1.);
}

template<class real>
void CUDABGFuncs<real>::calculate_E_2d(DeviceMatrix *E,
                                         const DeviceVector &b0, const DeviceVector &b1,
                                         const DeviceMatrix &W,
                                         const DeviceMatrix &x0, const DeviceMatrix &x1) {
    DeviceVector *bx0 = devMath.tempDeviceVector(x0.rows, __func__);
    DeviceVector *bx1 = devMath.tempDeviceVector(x1.rows, __func__);
    devMath.vmProduct(bx0, 1., b0, x0, opTranspose);
    devMath.vmProduct(bx1, 1., b1, x1, opTranspose);

    devMath.mmmProduct(E, 1., x1, opNone, W, opNone, x0, opTranspose);
    devMath.scaleBroadcast(E, 1., *bx0, opRowwise, 1.);
    devMath.scaleBroadcast(E, 1., *bx1, opColwise, 1.);
}


template<class real>
void CUDABGFuncs<real>::calculate_hJc(DeviceVector *h0, DeviceVector *h1,
                                        DeviceMatrix *J, DeviceScalar *c,
                                        const DeviceVector &b0, const DeviceVector &b1,
                                        const DeviceMatrix &W) {
    devMath.scale(J, 0.25, W);
    devMath.sumBatched(h0, 1., *J, opColwise);
    devMath.scale(h0, 0.5, b0, 1.);
    devMath.sumBatched(h1, 1., *J, opRowwise);
    devMath.scale(h1, 0.5, b1, 1.);

    devMath.sum(c, 1., *J);
    devMath.sum(c, 0.5, b0, 1.);
    devMath.sum(c, 0.5, b1, 1.);
}


template<class real>
void CUDABGFuncs<real>::calculate_E(DeviceScalar *E,
                                      const DeviceVector &h0, const DeviceVector &h1,
                                      const DeviceMatrix &J, const DeviceScalar &c,
                                      const DeviceVector &q0, const DeviceVector &q1) {
    devMath.vmvProduct(E, 1., q1, J, q0);
    devMath.dot(E, 1., h0, q0, 1.);
    devMath.dot(E, 1., h1, q1, 1.);
    devMath.scale(E, 1., c, 1.);
}


template<class real>
void CUDABGFuncs<real>::calculate_E(DeviceVector *E,
                                      const DeviceVector &h0, const DeviceVector &h1,
                                      const DeviceMatrix &J, const DeviceScalar &c,
                                      const DeviceMatrix &q0, const DeviceMatrix &q1) {
    devMath.vmvProductBatched(E, 1., q1, J, q0);
    devMath.vmProduct(E, 1., h0, q0, opTranspose, 1.);
    devMath.vmProduct(E, 1., h1, q1, opTranspose, 1.);
    devMath.scaleBroadcast(E, 1., c, 1.);
}

template<class real>
void CUDABGFuncs<real>::assignDevice(Device &device, DeviceStream *stream) {
    devMath.assignDevice(device, stream);
}


template struct CUDADGFuncs<double>;
template struct CUDADGFuncs<float>;
template struct CUDABGFuncs<double>;
template struct CUDABGFuncs<float>;
