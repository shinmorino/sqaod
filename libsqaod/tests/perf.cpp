#include <cpu/CPUDenseGraphBFSolver.h>
#include <cpu/CPUDenseGraphAnnealer.h>
#include <iostream>
#include <chrono>

namespace sq = sqaod;

#undef SQAOD_CUDA_ENABLED

#ifdef SQAOD_CUDA_ENABLED
#  include <cuda/CUDADenseGraphBFSolver.h>
#  include <cuda/CUDADenseGraphAnnealer.h>
namespace sqcuda = sqaod_cuda;
#endif

template<class T>
void showDuration(const T &duration) {
    std::cout << "elapsed time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " msec."
              << std::endl;
}


template<class real>
sq::MatrixType<real> symmetricMatrix(sq::SizeType dim) {
    sq::Random random;
    random.seed(0);
    sq::MatrixType<real> mat(dim, dim);
    for (sq::SizeType irow = 0; irow < dim; ++irow) {
        for (sq::SizeType icol = irow; icol < dim; ++icol) {
            mat(icol, irow) = mat(irow, icol) = random.random<real>() - 0.5f;
        }
    }
    return mat;
}

template<class real>
void denseGraphBFSearch(int N) {

    sq::MatrixType<real> W = symmetricMatrix<real>(N);

    sq::CPUDenseGraphBFSolver<real> cpuSolver;
    cpuSolver.setProblem(W);
    cpuSolver.setTileSize(1 << std::min(N, 18));

    auto start = std::chrono::system_clock::now();
    cpuSolver.search();
    auto end = std::chrono::system_clock::now();

    std::cout << cpuSolver.get_E().min() << std::endl;

    showDuration(end - start);

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

    showDuration(end - start);
#endif
}

template<class real, template<class real> class A>
void anneal(A<real> &an, real Ginit, real Gfin, real kT, real tau) {

    an.initAnneal();
    an.randomize_q();
    real G = Ginit;
    while (Gfin < G) {
        an.annealOneStep(G, kT);
        G = G * tau;
        std::cerr << ".";
    }
    an.finAnneal();
    std::cerr << std::endl;
    const sq::VectorType<real> &E = an.get_E();
    std::cerr << "Energy : " << E.min() << std::endl;
}

template<class real>
void denseGraphAnnealer(int N) {

    real Ginit = 5.;
    real Gfin = 0.01;
    real kT = 0.02;
    real tau = 0.99;

    sq::MatrixType<real> W = symmetricMatrix<real>(N);

    sq::CPUDenseGraphAnnealer<real> cpuAnnealer;
    cpuAnnealer.seed(0);
    cpuAnnealer.selectAlgorithm(sq::algoNaive);
    cpuAnnealer.setProblem(W);
    cpuAnnealer.setNumTrotters(N / 2);

    auto start = std::chrono::system_clock::now();
    anneal(cpuAnnealer, Ginit, Gfin, kT, tau);
    auto end = std::chrono::system_clock::now();

    std::cout << cpuAnnealer.get_E().min() << std::endl;

    showDuration(end - start);

#ifdef SQAOD_CUDA_ENABLED
    sqcuda::Device device;
    device.initialize();

    sqcuda::CUDADenseGraphAnnealer<real> cudaAnnealer(device);
    cudaAnnealer.setProblem(W);
    cudaAnnealer.setNumTrotters(N / 2);

    start = std::chrono::system_clock::now();
    anneal(cudaAnnealer, Ginit, Gfin, kT, tau);
    end = std::chrono::system_clock::now();

    std::cout << cudaAnnealer.get_E().min() << std::endl;
    device.finalize();

    showDuration(end - start);
#endif
}

int main() {
    // int N = 20;
    // denseGraphBFSearch<double>(N);
    // denseGraphBFSearch<float>(N);

    int N = 500;
    // denseGraphAnnealer<double>(N);
    denseGraphAnnealer<float>(N);
}
