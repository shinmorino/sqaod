#pragma once

#include <cuda/DeviceMath.h>

namespace sqaod_cuda {
    
template<class real>
struct DeviceDenseGraphFormulas {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceMathType<real> DeviceMath;
    
    void calculate_E(DeviceScalar *E, const DeviceMatrix &W, const DeviceVector &x);
    
    void calculate_E(DeviceVector *E, const DeviceMatrix &W, const DeviceMatrix &x);
    
    void calculate_hJc(DeviceVector *h, DeviceMatrix *J, DeviceScalar *c, const DeviceMatrix &W);
    
    void calculate_E(DeviceScalar *E,
                     const DeviceVector &h, const DeviceMatrix &J, const DeviceScalar &c,
                     const DeviceVector &q);

    void calculate_E(DeviceVector *E,
                     const DeviceVector &h, const DeviceMatrix &J, const DeviceScalar &c,
                     const DeviceMatrix &q);



    DeviceDenseGraphFormulas();

    DeviceDenseGraphFormulas(Device &device, DeviceStream *stream = NULL);
    void assignDevice(Device &device, DeviceStream *stream = NULL);

    DeviceMath devMath;
};


    
template<class real>
struct DeviceBipartiteGraphFormulas {
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    typedef DeviceMathType<real> DeviceMath;
    
    void calculate_E(DeviceScalar *E,
                     const DeviceVector &b0, const DeviceVector &b1, const DeviceMatrix &W,
                     const DeviceVector &x0, const DeviceVector &x1);
    
    void calculate_E(DeviceVector *E,
                     const DeviceVector &b0, const DeviceVector &b1, const DeviceMatrix &W,
                     const DeviceMatrix &x0, const DeviceMatrix &x1);

    void calculate_E_2d(DeviceMatrix *E,
                        const DeviceVector &b0, const DeviceVector &b1, const DeviceMatrix &W,
                        const DeviceMatrix &x0, const DeviceMatrix &x1);
    
    void calculate_hJc(DeviceVector *h0, DeviceVector *h1, DeviceMatrix *J, DeviceScalar *c,
                       const DeviceVector &b0, const DeviceVector &b1, const DeviceMatrix &W);

    void calculate_E(DeviceScalar *E,
                     const DeviceVector &h0, const DeviceVector &h1, const DeviceMatrix &J,
                     const DeviceScalar &c,
                     const DeviceVector &q0, const DeviceVector &q1);

    void calculate_E(DeviceVector *E,
                     const DeviceVector &h0, const DeviceVector &h1, const DeviceMatrix &J,
                     const DeviceScalar &c,
                     const DeviceMatrix &q0, const DeviceMatrix &q1);


    DeviceBipartiteGraphFormulas();

    DeviceBipartiteGraphFormulas(Device &device, DeviceStream *stream = NULL);

    void assignDevice(Device &device, DeviceStream *stream = NULL);

    DeviceMath devMath;
};

}
