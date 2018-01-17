#include <cuda/Device.h>
#include <cuda/DeviceMath.h>
#include <iostream>

using namespace sqaod_cuda;

template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceMatrixType<real> &dmat) {
    sqaod::MatrixType<real> hmat;
    DeviceCopyType<real>()(&hmat, dmat);
    ostm << hmat.map() << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceVectorType<real> &dvec) {
    sqaod::VectorType<real> hvec;
    DeviceCopyType<real>()(&hvec, dvec);
    ostm << hvec.mapToRowVector() << std::endl;
    return ostm;
}

template<class real>
std::ostream &operator<<(std::ostream &ostm, const DeviceScalarType<real> &ds) {
    real hs;
    DeviceCopyType<real>()(&hs, ds);
    ostm << hs << std::endl;
    return ostm;
}


void testMemstore() {
    Device device;
    device.initialize(0);
    
    DeviceObjectAllocatorType<double> *alloc = device.objectAllocator<double>();
    DeviceMatrixType<double> mat;
    alloc->allocate(&mat, 10, 10);
    alloc->deallocate(mat);

    DeviceVectorType<double> vec;
    alloc->allocate(&vec, 10);
    alloc->deallocate(vec);

    DeviceScalarType<double> sc;
    alloc->allocate(&sc);
    alloc->deallocate(sc);
    
    DeviceStream *defStream = device.defaultStream();

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
    DeviceCopyType<real> devCopy(device);
    DeviceStream *devStream = device.defaultStream();

    typedef sq::MatrixType<real> HostMatrix;
    typedef sq::VectorType<real> HostVector;
    typedef DeviceMatrixType<real> DeviceMatrix;
    typedef DeviceVectorType<real> DeviceVector;
    typedef DeviceScalarType<real> DeviceScalar;
    auto *alloc = device.objectAllocator<real>();
    
    DeviceMatrix dA, dB, dC, dD;
    DeviceVector dx, dy, dz;
    DeviceScalar da, db, dc;

    alloc->allocate(&dA, 5, 5);
    devCopy(&dA, 0.); /* create zero matrix */
    device.synchronize();
    std::cout << dA << std::endl;

    devMath.setToDiagonals(&dA, real(1.));
    device.synchronize();
    std::cout << dA << std::endl;

    devMath.scale(&dB, 10., dA);
    device.synchronize();
    std::cout << dB << std::endl;

    devMath.sum(&da, real(3.), dB);
    device.synchronize();
    std::cout << da << std::endl;

    const int nRows = 5, nCols = 5;
    HostMatrix hmat(nRows, nCols);
    for (int iRow = 0; iRow < nRows; ++iRow) {
        for (int iCol = 0; iCol < nCols; ++iCol) {
            hmat(iRow, iCol) = iRow * 10 + iCol;
        }
    }
    // std::cout << hmat.map() << std::endl;

    alloc->allocate(&dC, hmat.dim());
    devCopy(&dC, hmat);
    device.synchronize();
    std::cout << dC << std::endl;
    devMath.transpose(&dD, dC);
    device.synchronize();
    std::cout << dD << std::endl;


    devMath.sumBatched(&dx, 3., dA, opColwise);
    device.synchronize();
    std::cout << dx << std::endl;
    devMath.sumBatched(&dy, 3., dB, opRowwise);
    device.synchronize();
    std::cout << dy << std::endl;
    devMath.dot(&da, 3., dx, dy);
    device.synchronize();
    std::cout << da << std::endl;

    device.synchronize();

    alloc->deallocate(dA);
    alloc->deallocate(dB);
    alloc->deallocate(dx);
    alloc->deallocate(dy);
    alloc->deallocate(da);

    device.finalize();
    
};


int main() {
    // testMemstore();
    testDeviceMath<double>();
    // testDeviceMath<float>();
}
