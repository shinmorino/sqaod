#include <cpu/CPUDenseGraphBFSolver.h>
#include <cpu/CPURandom.h>
#include <iostream>
#include <chrono>

namespace sq = sqaod;


#ifdef SQAOD_CUDA_ENABLED
#  include <cuda/CUDADenseGraphBFSolver.h>
namespace sqcuda = sqaod_cuda;
#endif


template<class real>
sq::MatrixType<real> symmetricMatrix(sq::SizeType dim) {
    sq::CPURandom random;
    random.seed(0);
    sq::MatrixType<real> mat(dim, dim);
    for (sq::SizeType irow = 0; irow < dim; ++irow) {
        for (sq::SizeType icol = irow; icol < dim; ++icol) {
            mat(icol, irow) = mat(irow, icol) = random.random<real>() - 0.5f;
        }
    }
    return mat;
}

int main() {

    typedef double real;

    int N = 20;

    sq::MatrixType<real> W = symmetricMatrix<real>(N);
    
    sq::CPUDenseGraphBFSolver<real> cpuSolver;
    cpuSolver.setProblem(W);
    cpuSolver.setTileSize(1 << std::min(N, 18));

    auto start = std::chrono::system_clock::now();
    cpuSolver.search();
    auto end = std::chrono::system_clock::now();

    std::cout << cpuSolver.get_E().min() << std::endl;

    auto diff = end - start;
    std::cout << "elapsed time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " msec."
              << std::endl;

#ifdef SQAOD_CUDA_ENABLED    
    sqcuda::Device device;
    device.initialize();

    sqcuda::CUDADenseGraphBFSolver<real> cudaSolver(device);
    cudaSolver.setProblem(W);
    cudaSolver.setTileSize(1 << std::min(N, 18));

    start = std::chrono::system_clock::now();
    cudaSolver.search();
    end = std::chrono::system_clock::now();

    std::cout << cudaSolver.get_E().min() << std::endl;
    device.finalize();

    diff = end - start;
    std::cout << "elapsed time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " msec."
              << std::endl;
#endif
}
