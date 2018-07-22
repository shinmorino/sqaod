#include <sqaodc/sqaodc_native.h>
#include <iostream>
#include <chrono>

namespace sqn = sqaod::native;
namespace sq = sqaod;

bool runFloatSolvers = true;
bool runDoubleSolvers = true;

bool runDenseGraphBruteForceSearchers = true;
bool runDenseGraphAnnealers = true;
bool runBipartiteGraphBruteForceSearchers = true;
bool runBipartiteGraphAnnealers = true;

bool runCPUSolvers = true;
bool runCUDASolvers = true;

const int nSteps = 200;

const int SEED = 1133557;

#ifdef SQAODC_CUDA_ENABLED
sqaod_cuda::Device device;
#endif

template<class T>
void showDuration(const T &duration) {
    std::cout << "elapsed time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " msec."
              << std::endl;
}

template<class real>
sq::MatrixType<real> symmetricMatrix(sq::SizeType dim) {
    sq::MatrixType<double> mat(dim, dim);
    for (sq::SizeType irow = 0; irow < dim; ++irow) {
        for (sq::SizeType icol = irow; icol < dim; ++icol) {
            mat(icol, irow) = mat(irow, icol) = sq::random.random<double>() - 0.5;
        }
    }
    return sq::cast<real>(mat);
}

template<class real>
sq::MatrixType<real> matrix(const sq::Dim &dim) {
    sq::MatrixType<double> mat(dim.rows, dim.cols);
    for (sq::SizeType irow = 0; irow < dim.rows; ++irow) {
        for (sq::SizeType icol = 0; icol < dim.cols; ++icol) {
            mat(irow, icol) = sq::random.random<double>() - 0.5;
        }
    }
    return sq::cast<real>(mat);
}


template<class real>
sq::VectorType<real> vector(sq::SizeType size) {
    sq::VectorType<double> vec(size);
    for (sq::SizeType idx = 0; idx < size; ++idx) {
        vec(idx) = sq::random.random<double>() - 0.5;
    }
    return sq::cast<real>(vec);
}

template<class real>
void createBipartiteGraph(sq::VectorType<real> *b0, sq::VectorType<real> *b1, sq::MatrixType<real> *W,
                          sq::SizeType N0, sq::SizeType N1) {
    *b0 = vector<real>(N0);
    *b1 = vector<real>(N1);
    *W = matrix<real>(sq::Dim(N1, N0));
}

template<class real, template<class> class S>
void runSearch(S<real> &searcher) {
    auto start = std::chrono::system_clock::now();
    searcher.search();
    auto end = std::chrono::system_clock::now();

    const sq::VectorType<real> &E = searcher.get_E();
    std::cerr << "Energy : " << E.min() << std::endl;
    showDuration(end - start);
    std::cerr << std::endl;
}


template<class real, template<class> class A>
void anneal(A<real> &an) {
    real Ginit = real(20.);
    real Gfin = real(0.01);
    real beta = real(1.) / real(0.02);
    real tau = std::exp(std::log(Ginit / Gfin) / nSteps);

    auto start = std::chrono::system_clock::now();
    an.prepare();
    an.randomizeSpin();
    real G = Ginit;
    for (int idx = 0; idx < nSteps; ++idx) {
        an.annealOneStep(G, beta);
        G = G * tau;
        std::cerr << ".";
    }
    an.makeSolution();
    auto end = std::chrono::system_clock::now();

    std::cerr << std::endl;
    const sq::VectorType<real> &E = an.get_E();
    std::cerr << "Energy  : " << E.min() << std::endl;
    std::cerr << "# Steps : " << nSteps << std::endl;
    showDuration(end - start);
    std::cerr << std::endl;
}

template<class real>
void run(const char *precisionStr) {
    /* Dense graph brute-force searchers */
    if (runDenseGraphBruteForceSearchers) {
        int N = 24;
        sq::random.seed(SEED);
        sq::MatrixType<real> W = symmetricMatrix<real>(N);
        if (runCPUSolvers) {
            fprintf(stderr, "Dense graph brute-force searcher, CPU, %s\n", precisionStr);
            fprintf(stderr, "N = %d\n", N);
            sqn::cpu::DenseGraphBFSearcher<real> searcher;
            searcher.setQUBO(W);
            runSearch(searcher);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Dense graph brute-force searcher, CUDA, %s\n", precisionStr);
            fprintf(stderr, "N = %d\n", N);
            sqn::cuda::DenseGraphBFSearcher<real> searcher(device);
            searcher.setQUBO(W);
            runSearch(searcher);
        }
#endif
    }

    /* Dense graph annealers */
    if (runDenseGraphAnnealers) {
        int N = 1024;
        int m = N  / 2;
        sq::random.seed(SEED);
        sq::MatrixType<real> W = symmetricMatrix<real>(N);
        if (runCPUSolvers) {
            fprintf(stderr, "Dense graph annealer, CPU, %s\n", precisionStr);
            fprintf(stderr, "N = %d, m = %d\n", N, m);
            sqn::cpu::DenseGraphAnnealer<real> annealer;
            annealer.seed(SEED);
            annealer.setQUBO(W);
            annealer.setPreference(sq::pnNumTrotters, N / 2);
            anneal(annealer);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Dense graph annealer, CUDA, %s\n", precisionStr);
            fprintf(stderr, "N = %d, m = %d\n", N, m);
            sqn::cuda::DenseGraphAnnealer<real> annealer(device);
            annealer.seed(SEED);
            annealer.setQUBO(W);
            annealer.setPreference(sq::pnNumTrotters, N / 2);
            anneal(annealer);
        }
#endif
    }

    /* Bipartite graph brute-force searchers */
    if (runBipartiteGraphBruteForceSearchers) {
        int N0 = 14, N1 = 14;
        sq::random.seed(SEED);
        sq::VectorType<real> b0, b1;
        sq::MatrixType<real> W;
        createBipartiteGraph(&b0, &b1, &W, N0, N1);
        if (runCPUSolvers) {
            fprintf(stderr, "Bipartite graph brute-force searcher, CPU, %s\n", precisionStr);
            fprintf(stderr, "(N0, N1) = (%d, %d)\n", N0, N1);
            sqn::cpu::BipartiteGraphBFSearcher<real> searcher;
            searcher.setQUBO(b0, b1, W);
            runSearch(searcher);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Bipartite graph brute-force searcher, CUDA, %s\n", precisionStr);
            fprintf(stderr, "(N0, N1) = (%d, %d)\n", N0, N1);
            sqn::cuda::BipartiteGraphBFSearcher<real> searcher(device);
            searcher.setQUBO(b0, b1, W);
            runSearch(searcher);
        }
#endif
    }

    /* Bipartite graph annealers */
    if (runBipartiteGraphAnnealers) {
        int N0 = 1024, N1 = 512;
        int m = (N0 + N1) / 2;
        sq::random.seed(SEED);
        sq::VectorType<real> b0, b1;
        sq::MatrixType<real> W;
        createBipartiteGraph(&b0, &b1, &W, N0, N1);
        if (runCPUSolvers) {
            fprintf(stderr, "Bipartite graph annealer, CPU, %s\n", precisionStr);
            fprintf(stderr, "(N0, N1) = (%d, %d), m = %d\n", N0, N1, m);
            sqn::cpu::BipartiteGraphAnnealer<real> annealer;
            annealer.seed(SEED);
            annealer.setQUBO(b0, b1, W);
            annealer.setPreference(sq::pnNumTrotters, sq::SizeType((N0 + N1) / 2));
            anneal(annealer);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Bipartite graph annealer, CUDA, %s\n", precisionStr);
            fprintf(stderr, "(N0, N1) = (%d, %d), m = %d\n", N0, N1, m);
            sqn::cuda::BipartiteGraphAnnealer<real> annealer(device);
            annealer.seed(SEED);
            annealer.setQUBO(b0, b1, W);
            annealer.setPreference(sq::pnNumTrotters, (N0 + N1) / 2);
            anneal(annealer);
        }
#endif
    }
}

int main() {

#ifdef SQAODC_CUDA_ENABLED
    if (sq::isCUDAAvailable())
        device.initialize();
    else
        runCUDASolvers = false;
#else
    runCUDASolvers = false;
#endif

    if (runFloatSolvers)
        run<float>("float");
    if (runDoubleSolvers)
        run<double>("double");

#ifdef SQAODC_CUDA_ENABLED
    if (sq::isCUDAAvailable()) {
        device.finalize();
        cudaDeviceReset();
    }
#endif
}
