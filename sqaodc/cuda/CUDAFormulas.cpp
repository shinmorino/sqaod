#include "CUDAFormulas.h"
#include <float.h>


using namespace sqaod_cuda;

template<class real> void CUDADenseGraphFormulas<real>::
calculate_E(real *E, const HostMatrix &W, const HostVector &x) {
    DeviceScalar *d_E = devStream->tempDeviceScalar<real>();
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceVector *d_x = devStream->tempDeviceVector<real>(x.size);
    devCopy(d_W, W);
    devCopy(d_x, x);
    formulas.calculate_E(d_E, *d_W, *d_x);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real> void CUDADenseGraphFormulas<real>::
calculate_E(HostVector *E, const HostMatrix &W, const HostMatrix &x) {
    DeviceVector *d_E = devStream->tempDeviceVector<real>(x.rows);
    DeviceMatrix *d_W = devStream->tempDeviceMatrix<real>(W.dim());
    DeviceMatrix *d_x = devStream->tempDeviceMatrix<real>(x.dim());
    devCopy(d_W, W);
    devCopy(d_x, x);
    formulas.calculate_E(d_E, *d_W, *d_x);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real> void CUDADenseGraphFormulas<real>::
calculateHamiltonian(HostVector *h, HostMatrix *J, real *c, const HostMatrix &W) {
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

template<class real> void CUDADenseGraphFormulas<real>::
calculate_E(real *E,
            const HostVector &h, const HostMatrix &J, const real &c,
            const HostVector &q) {
    DeviceScalar *d_E = devStream->tempDeviceScalar<real>();
    DeviceVector *d_h = devStream->tempDeviceVector<real>(h.size);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(J.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceVector *d_q = devStream->tempDeviceVector<real>(q.size);
    formulas.calculate_E(d_E, *d_h, *d_J, *d_c, *d_q);
    devCopy(E, *d_E);
    devStream->synchronize();
}

template<class real> void CUDADenseGraphFormulas<real>::
calculate_E(HostVector *E,
            const HostVector &h, const HostMatrix &J, const real &c,
            const HostMatrix &q) {
    DeviceVector *d_E = devStream->tempDeviceVector<real>(q.rows);
    DeviceVector *d_h = devStream->tempDeviceVector<real>(h.size);
    DeviceMatrix *d_J = devStream->tempDeviceMatrix<real>(J.dim());
    DeviceScalar *d_c = devStream->tempDeviceScalar<real>();
    DeviceMatrix *d_q = devStream->tempDeviceMatrix<real>(q.dim());
    formulas.calculate_E(d_E, *d_h, *d_J, *d_c, *d_q);
    devCopy(E, *d_E);
    devStream->synchronize();
}


template<class real>
CUDADenseGraphFormulas<real>::CUDADenseGraphFormulas() {
    devStream = NULL;
}


template<class real>
CUDADenseGraphFormulas<real>::CUDADenseGraphFormulas(Device &device, DeviceStream *stream) {
    devStream = NULL;
    assignDevice(device, stream);
}

template<class real>
void CUDADenseGraphFormulas<real>::assignDevice(Device &device, DeviceStream *stream) {
    throwErrorIf(devStream != NULL, "Device already assigned.");
    stream = stream;
    devCopy.assignDevice(device, stream);
    formulas.assignDevice(device, stream);
}


/* Bipartite graph */

template<class real> void CUDABipartiteGraphFormulas<real>::
calculate_E(real *E,
            const HostVector &b0, const HostVector &b1, const HostMatrix &W,
            const HostVector &x0, const HostVector &x1) {
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

template<class real> void CUDABipartiteGraphFormulas<real>::
calculate_E(HostVector *E,
            const HostVector &b0, const HostVector &b1, const HostMatrix &W,
            const HostMatrix &x0, const HostMatrix &x1) {
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
void CUDABipartiteGraphFormulas<real>::
calculate_E_2d(HostMatrix *E,
               const HostVector &b0, const HostVector &b1, const HostMatrix &W,
               const HostMatrix &x0, const HostMatrix &x1) {
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


template<class real> void CUDABipartiteGraphFormulas<real>::
calculateHamiltonian(HostVector *h0, HostVector *h1, HostMatrix *J, real *c,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W) {
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
void CUDABipartiteGraphFormulas<real>::
calculate_E(real *E,
            const HostVector &h0, const HostVector &h1, const HostMatrix &J, const real &c,
            const HostVector &q0, const HostVector &q1) {
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

template<class real> void CUDABipartiteGraphFormulas<real>::
calculate_E(HostVector *E,
            const HostVector &h0, const HostVector &h1, const HostMatrix &J, const real &c,
            const HostMatrix &q0, const HostMatrix &q1) {
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
CUDABipartiteGraphFormulas<real>::CUDABipartiteGraphFormulas() {
    devStream = NULL;
}

template<class real>
CUDABipartiteGraphFormulas<real>::CUDABipartiteGraphFormulas(Device &device, DeviceStream *stream) {
    devStream = NULL;
    assignDevice(device, stream);
}

template<class real>
void CUDABipartiteGraphFormulas<real>::assignDevice(Device &device, DeviceStream *stream) {
    throwErrorIf(devStream != NULL, "Device already assigned.");
    devStream = stream;
    devCopy.assignDevice(device, stream);
    formulas.assignDevice(device, stream);
}



template struct CUDADenseGraphFormulas<double>;
template struct CUDADenseGraphFormulas<float>;
template struct CUDABipartiteGraphFormulas<double>;
template struct CUDABipartiteGraphFormulas<float>;
