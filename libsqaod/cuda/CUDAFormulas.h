#pragma once

#include <cuda/DeviceFormulas.h>

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
    
    void calculate_E(real *E, const HostMatrix &W, const HostVector &x);
    
    void calculate_E(HostVector *E, const HostMatrix &W, const HostMatrix &x);
    
    void calculate_hJc(HostVector *h, HostMatrix *J, real *c, const HostMatrix &W);
    
    void calculate_E(real *E,
                     const HostVector &h, const HostMatrix &J, const real &c,
                     const HostVector &q);

    void calculate_E(HostVector *E,
                     const HostVector &h, const HostMatrix &J, const real &c,
                     const HostMatrix &q);


    CUDADenseGraphFormulas();

    CUDADenseGraphFormulas(Device &device, DeviceStream *stream = NULL);

    void assignDevice(Device &device, DeviceStream *stream = NULL);

    DeviceStream *devStream;
    DeviceCopy devCopy;
    DeviceFormulas formulas;
};


    
template<class real>
struct CUDABipartiteGraphFormulas {
    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceBipartiteGraphFormulas<real> DeviceFormulas;
    
    void calculate_E(real *E,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                     const HostVector &x0, const HostVector &x1);
    
    void calculate_E(HostVector *E,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                     const HostMatrix &x0, const HostMatrix &x1);

    void calculate_E_2d(HostMatrix *E,
                        const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                        const HostMatrix &x0, const HostMatrix &x1);
    
    void calculate_hJc(HostVector *h0, HostVector *h1, HostMatrix *J, real *c,
                       const HostVector &b0, const HostVector &b1, const HostMatrix &W);

    void calculate_E(real *E,
                     const HostVector &h0, const HostVector &h1, const HostMatrix &J,
                     const real &c,
                     const HostVector &q0, const HostVector &q1);

    void calculate_E(HostVector *E,
                     const HostVector &h0, const HostVector &h1, const HostMatrix &J,
                     const real &c,
                     const HostMatrix &q0, const HostMatrix &q1);

    CUDABipartiteGraphFormulas();

    CUDABipartiteGraphFormulas(Device &device, DeviceStream *stream = NULL);

    void assignDevice(Device &device, DeviceStream *stream = NULL);

    DeviceStream *devStream;
    DeviceCopy devCopy;
    DeviceFormulas formulas;
};

}
