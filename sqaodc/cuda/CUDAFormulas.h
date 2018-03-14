#pragma once

#include <sqaodc/cuda/DeviceFormulas.h>

namespace sqaod_cuda {

namespace sq = sqaod;

template<class real>
struct CUDADenseGraphFormulas {
    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceDenseGraphFormulas<real> DeviceFormulas;
    
    static
    void calculate_E(real *E, const HostMatrix &W, const HostVector &x);
    
    static
    void calculate_E(HostVector *E, const HostMatrix &W, const HostMatrix &x);
    
    static
    void calculateHamiltonian(HostVector *h, HostMatrix *J, real *c, const HostMatrix &W);
    
    static
    void calculate_E(real *E,
                     const HostVector &h, const HostMatrix &J, const real &c,
                     const HostVector &q);

    static
    void calculate_E(HostVector *E,
                     const HostVector &h, const HostMatrix &J, const real &c,
                     const HostMatrix &q);


    static
    void assignDevice(Device &device, DeviceStream *stream = NULL);

    static
    DeviceStream *devStream;
    static
    DeviceCopy devCopy;
    static
    DeviceFormulas formulas;

private:
    
    CUDADenseGraphFormulas();
};


    
template<class real>
struct CUDABipartiteGraphFormulas {
    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceBipartiteGraphFormulas<real> DeviceFormulas;
    
    static
    void calculate_E(real *E,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                     const HostVector &x0, const HostVector &x1);
    
    static
    void calculate_E(HostVector *E,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                     const HostMatrix &x0, const HostMatrix &x1);

    static
    void calculate_E_2d(HostMatrix *E,
                        const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                        const HostMatrix &x0, const HostMatrix &x1);
    
    static
    void calculateHamiltonian(HostVector *h0, HostVector *h1, HostMatrix *J, real *c,
                              const HostVector &b0, const HostVector &b1, const HostMatrix &W);

    static
    void calculate_E(real *E,
                     const HostVector &h0, const HostVector &h1, const HostMatrix &J,
                     const real &c,
                     const HostVector &q0, const HostVector &q1);

    static
    void calculate_E(HostVector *E,
                     const HostVector &h0, const HostVector &h1, const HostMatrix &J,
                     const real &c,
                     const HostMatrix &q0, const HostMatrix &q1);

    static
    void assignDevice(Device &device, DeviceStream *stream = NULL);

    static
    DeviceStream *devStream;
    static
    DeviceCopy devCopy;
    static
    DeviceFormulas formulas;

private:
    CUDABipartiteGraphFormulas();
};

}
