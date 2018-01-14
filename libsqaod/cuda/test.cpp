#include <cuda/Device.h>
#include <cuda/DeviceMath.h>

using namespace sqaod_cuda;

void testMemstore() {
    Device device;
    device.initialize(0);
    
    DeviceObjectAllocatorType<double> *alloc = device.deviceObjectAllocator<double>();
    DeviceMatrixType<double> mat;
    alloc->allocate(&mat, 10, 10);
    alloc->deallocate(mat);

    DeviceVectorType<double> vec;
    alloc->allocate(&vec, 10);
    alloc->deallocate(vec);

    DeviceScalarType<double> sc;
    alloc->allocate(&sc);
    alloc->deallocate(sc);
    
    DeviceStream *defStream = device.defaultDeviceStream();

    DeviceMatrixType<double> *tmpMat = defStream->tempDeviceMatrix<double>(10, 10);
    DeviceVectorType<double> *tmpVec = defStream->tempDeviceVector<double>(10);
    DeviceScalarType<double> *tmpSc = defStream->tempDeviceScalar<double>();
    
    defStream->synchronize();
    
    device.finalize();
}

template<class real>
void testDeviceMath() {

    namespace sq = sqaod;

    Device device;
    device.initialize(0);
    DeviceMathType<real> devMath(device);
    DeviceCopy devCopy; //(device);
    DeviceStream *devStream = device.defaultDeviceStream();
    

    typedef sq::MatrixType<real> HostMatrix;
    typedef DeviceMatrixType<real> DeviceMatrix;
    
    DeviceMatrix dA, dB, dC;

    HostMatrix hMat = HostMatrix::zeros(10, 10);
    devCopy(&dA, hMat);
    devMath.setToDiagonals(&dA, real(1.));
    devStream->synchronize();
    
    devMath.scale(&dB, 10., dA);
    devStream->synchronize();
    
    
    
};


int main() {
    testMemstore();
    testDeviceMath<double>();
    testDeviceMath<float>();
}
