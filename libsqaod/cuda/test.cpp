#include <cuda/Device.h>

using namespace sqaod_cuda;

int main() {
    Device device;
    device.initialize(0);
    
    DeviceObjectAllocatorType<double> &alloc = device.deviceObjectAllocator<double>();
    DeviceMatrixType<double> mat;
    alloc.allocate(&mat, 10, 10);
    alloc.deallocate(mat);

    DeviceVectorType<double> vec;
    alloc.allocate(&vec, 10);
    alloc.deallocate(vec);

    DeviceScalarType<double> sc;
    alloc.allocate(&sc);
    alloc.deallocate(sc);
    
    DeviceStream &defStream = device.defaultDeviceStream();

    DeviceMatrixType<double> *tmpMat = defStream.tempDeviceMatrix<double>(10, 10);
    DeviceVectorType<double> *tmpVec = defStream.tempDeviceVector<double>(10);
    DeviceScalarType<double> *tmpSc = defStream.tempDeviceScalar<double>();
    
    defStream.synchronize();
    
    device.finalize();
}
