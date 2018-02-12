#include <cpu/CPUDenseGraphBFSolver.h>
#include <cpu/CPUDenseGraphAnnealer.h>
#include <cpu/CPUBipartiteGraphBFSolver.h>
#include <cpu/CPUBipartiteGraphAnnealer.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

namespace sq = sqaod;

#ifdef SQAOD_CUDA_ENABLED
#  include <cuda/CUDADenseGraphBFSolver.h>
#  include <cuda/CUDADenseGraphAnnealer.h>
#  include <cuda/CUDABipartiteGraphBFSolver.h>
#  include <cuda/CUDABipartiteGraphAnnealer.h>
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
sq::MatrixType<real> matrix(const sq::Dim &dim) {
    sq::MatrixType<real> mat(dim.rows, dim.cols);
    for (sq::SizeType irow = 0; irow < dim.rows; ++irow) {
        for (sq::SizeType icol = 0; icol < dim.cols; ++icol) {
            mat(irow, icol) = sq::random.random<real>() - 0.5f;
        }
    }
    return mat;
}


template<class real>
sq::VectorType<real> vector(sq::SizeType size) {
    sq::VectorType<real> vec(size);
    for (sq::SizeType idx = 0; idx < size; ++idx) {
        vec(idx) = sq::random.random<real>() - 0.5f;
    }
    return vec;
}

template<class real>
void denseGraphBFSearch(int N) {
    sq::MatrixType<real> W = symmetricMatrix<real>(N);

    sq::CPUDenseGraphBFSolver<real> cpuSolver;
    cpuSolver.setProblem(W);
    cpuSolver.setTileSize(1 << std::min(N, 20));

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

template<class real>
void bipartiteGraphBFSearch(int N0, int N1) {
    sq::VectorType<real> b0 = vector<real>(N0);
    sq::VectorType<real> b1 = vector<real>(N1);
    sq::MatrixType<real> W = matrix<real>(sq::Dim(N1, N0));

    sq::CPUBipartiteGraphBFSolver<real> cpuSolver;
    cpuSolver.setProblem(b0, b1, W);
//    cpuSolver.setTileSize(1 << std::min(N, 20));

    auto start = std::chrono::system_clock::now();
    cpuSolver.search();
    auto end = std::chrono::system_clock::now();

    std::cout << cpuSolver.get_E().min() << std::endl;

    showDuration(end - start);

#ifdef SQAOD_CUDA_ENABLED
    sqcuda::Device device;
    // device.useManagedMemory(true);
    // device.enableLocalStore(false);
    device.initialize();

    sqcuda::CUDABipartiteGraphBFSolver<real> cudaSolver(device);
    cudaSolver.setProblem(b0, b1, W);
    // cudaSolver.setTileSize(1 << std::min(N, 20));

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
    {
        sqcuda::CUDADenseGraphAnnealer<real> cudaAnnealer(device);
        cudaAnnealer.setProblem(W);
        cudaAnnealer.setNumTrotters(N / 2);
        cudaAnnealer.seed(1);

        auto start = std::chrono::system_clock::now();
        anneal(cudaAnnealer, Ginit, Gfin, kT, tau);
        auto end = std::chrono::system_clock::now();
        device.synchronize();
        std::cout << cudaAnnealer.get_E().min() << std::endl;
        showDuration(end - start);
    }
    device.finalize();

#endif
}


template<class real>
void bipartiteGraphAnnealer(int N0, int N1) {

    real Ginit = 5.;
    real Gfin = 0.01;
    real kT = 0.02;
    real tau = 0.99;

    sq::VectorType<real> b0 = vector<real>(N0);
    sq::VectorType<real> b1 = vector<real>(N1);
    sq::MatrixType<real> W = matrix<real>(sq::Dim(N1, N0));

    sq::CPUBipartiteGraphAnnealer<real> cpuAnnealer;
    cpuAnnealer.seed(0);
    // cpuAnnealer.selectAlgorithm(sq::algoNaive);
    cpuAnnealer.setProblem(b0, b1, W);
    cpuAnnealer.setNumTrotters((N0 + N1) / 2);

    auto start = std::chrono::system_clock::now();
    anneal(cpuAnnealer, Ginit, Gfin, kT, tau);
    auto end = std::chrono::system_clock::now();

    std::cout << cpuAnnealer.get_E().min() << std::endl;

    showDuration(end - start);

#ifdef SQAOD_CUDA_ENABLED
    sqcuda::Device device;
    device.initialize();
    {
        sqcuda::CUDABipartiteGraphAnnealer<real> cudaAnnealer(device);
        cudaAnnealer.setProblem(b0, b1, W);
        cudaAnnealer.setNumTrotters((N0 + N1) / 2);
        cudaAnnealer.seed(65743);

        auto start = std::chrono::system_clock::now();
        anneal(cudaAnnealer, Ginit, Gfin, kT, tau);
        auto end = std::chrono::system_clock::now();
        device.synchronize();
        std::cout << cudaAnnealer.get_E().min() << std::endl;
        showDuration(end - start);
    }
    device.finalize();

#endif
}



int main() {
    sq::random.seed(0);

    int N = 24;
    denseGraphBFSearch<double>(N);
    denseGraphBFSearch<float>(N);

    N = 1024;
    denseGraphAnnealer<double>(N);
    denseGraphAnnealer<float>(N);

    int N0 = 14, N1 = 14;
    bipartiteGraphBFSearch<float>(N0, N1);
    bipartiteGraphBFSearch<double>(N0, N1);

    bipartiteGraphAnnealer<float>(N0, N1);
    bipartiteGraphAnnealer<double>(N0, N1);
    
    cudaDeviceReset();
}
