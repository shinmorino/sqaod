#include <iostream>
#include "CUDADenseGraphBFSearcher.h"
#include "Device.h"

using namespace sqaod_cuda;
namespace sq = sqaod;


template<class real>
void runBFSearcher() {
    typedef sq::MatrixType<real> Matrix;

    Device device;
    device.initialize();
    
    int N = 100;
    CUDADenseGraphBFSearcher<real> solver;
    solver.assignDevice(device);
    Matrix W = Matrix::eye(N);
    solver.setProblem(W, sq::optMinimize);
    solver.search();

    device.finalize();

}


int main() {
    runBFSearcher<float>();
    runBFSearcher<double>();
}
