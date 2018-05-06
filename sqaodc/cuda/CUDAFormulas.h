#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/cuda/DeviceFormulas.h>

namespace sqaod_cuda {

template<class real>
struct CUDADenseGraphFormulas : sqaod::cuda::DenseGraphFormulas<real> {
    typedef sqaod::MatrixType<real> HostMatrix;
    typedef sqaod::VectorType<real> HostVector;
    typedef sqaod_cuda::DeviceMatrixType<real> DeviceMatrix;
    typedef sqaod_cuda::DeviceVectorType<real> DeviceVector;
    typedef sqaod_cuda::DeviceScalarType<real> DeviceScalar;
    typedef sqaod_cuda::DeviceDenseGraphFormulas<real> DeviceFormulas;
    
    void calculate_E(real *E, const HostMatrix &W, const HostVector &x);
    
    void calculate_E(HostVector *E, const HostMatrix &W, const HostMatrix &x);
    
    void calculateHamiltonian(HostVector *h, HostMatrix *J, real *c, const HostMatrix &W);
    
    void calculate_E(real *E,
                     const HostVector &h, const HostMatrix &J, real c,
                     const HostVector &q);

    void calculate_E(HostVector *E,
                     const HostVector &h, const HostMatrix &J, real c,
                     const HostMatrix &q);


    void assignDevice(sqaod::cuda::Device &device);

    sqaod_cuda::DeviceStream *devStream;
    sqaod_cuda::DeviceCopy devCopy;
    DeviceFormulas formulas;

    CUDADenseGraphFormulas();
    virtual ~CUDADenseGraphFormulas() { }
    
private:
    CUDADenseGraphFormulas(const CUDADenseGraphFormulas &);
};


    
template<class real>
struct CUDABipartiteGraphFormulas : sqaod::cuda::BipartiteGraphFormulas<real> {
    typedef sqaod::MatrixType<real> HostMatrix;
    typedef sqaod::VectorType<real> HostVector;
    typedef sqaod_cuda::DeviceMatrixType<real> DeviceMatrix;
    typedef sqaod_cuda::DeviceVectorType<real> DeviceVector;
    typedef sqaod_cuda::DeviceScalarType<real> DeviceScalar;
    typedef sqaod_cuda::DeviceBipartiteGraphFormulas<real> DeviceFormulas;

    void calculate_E(real *E,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                     const HostVector &x0, const HostVector &x1);
    
    void calculate_E(HostVector *E,
                     const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                     const HostMatrix &x0, const HostMatrix &x1);

    void calculate_E_2d(HostMatrix *E,
                        const HostVector &b0, const HostVector &b1, const HostMatrix &W,
                        const HostMatrix &x0, const HostMatrix &x1);
    
    void calculateHamiltonian(HostVector *h0, HostVector *h1, HostMatrix *J, real *c,
                              const HostVector &b0, const HostVector &b1, const HostMatrix &W);

    void calculate_E(real *E,
                     const HostVector &h0, const HostVector &h1, const HostMatrix &J,
                     real c,
                     const HostVector &q0, const HostVector &q1);

    void calculate_E(HostVector *E,
                     const HostVector &h0, const HostVector &h1, const HostMatrix &J,
                     real c,
                     const HostMatrix &q0, const HostMatrix &q1);

    void assignDevice(sqaod::cuda::Device &device);

    sqaod_cuda::DeviceStream *devStream;

    sqaod_cuda::DeviceCopy devCopy;

    DeviceFormulas formulas;

    CUDABipartiteGraphFormulas();
    virtual ~CUDABipartiteGraphFormulas() { }
    
private:
    CUDABipartiteGraphFormulas(const CUDABipartiteGraphFormulas &);
};

}
