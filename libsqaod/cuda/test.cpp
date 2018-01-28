#include <iostream>
#include "CUDADenseGraphBFSolver.h"
#include "Device.h"

using namespace sqaod_cuda;
namespace sq = sqaod;


template<class real>
void runBFSolver() {
    typedef sq::MatrixType<real> Matrix;

    Device device;
    device.initialize();
    
    int N = 100;
    CUDADenseGraphBFSolver<real> solver;
    solver.assignDevice(device);
    Matrix W = Matrix::eye(N);
    solver.setProblem(W, sq::optMinimize);
    solver.search();

    device.finalize();

}


int main() {
    runBFSolver<float>();
    runBFSolver<double>();
}
