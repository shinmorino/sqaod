#include <iostream>
#include "CUDADenseGraphBFSearcher.h"
#include "Device.h"

using namespace sqaod_cuda;


template<class real>
void runBFSearcher() {
    typedef sq::MatrixType<real> Matrix;

    Device device;
    device.initialize();
    
    int N = 100;
    CUDADenseGraphBFSearcher<real> solver;
    solver.assignDevice(device);
    Matrix W = Matrix::eye(N);
    solver.setQUBO(W, sq::optMinimize);
    solver.search();

    device.finalize();

}


int main() {
    runBFSearcher<float>();
    runBFSearcher<double>();
}
