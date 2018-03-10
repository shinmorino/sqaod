#include <sqaodc/sqaodc.h>
#include <iostream>
#include <chrono>

namespace sq = sqaod;

#ifdef SQAODC_CUDA_ENABLED
sq::cuda::Device device;
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
void createBipartiteGraph(sq::VectorType<real> *b0, sq::VectorType<real> *b1, sq::MatrixType<real> *W,
                          sq::SizeType N0, sq::SizeType N1) {
    *b0 = vector<real>(N0);
    *b1 = vector<real>(N1);
    *W = matrix<real>(sq::Dim(N1, N0));
}

template<class real, template<class real> class S>
void runSearch(S<real> &searcher) {
    auto start = std::chrono::system_clock::now();
    searcher.search();
    auto end = std::chrono::system_clock::now();

    const sq::VectorType<real> &E = searcher.get_E();
    std::cerr << "Energy : " << E.min() << std::endl;
    showDuration(end - start);
    std::cerr << std::endl;
}


template<class real, template<class real> class A>
void anneal(A<real> &an) {
    real Ginit = real(5.);
    real Gfin = real(0.01);
    real kT = real(0.02);
    real tau = real(0.99);
    tau = (real)0.9;

    int nSteps = 0;

    auto start = std::chrono::system_clock::now();
    an.prepare();
    an.randomize_q();
    real G = Ginit;
    while (Gfin < G) {
        an.annealOneStep(G, kT);
        G = G * tau;
        ++nSteps;
        std::cerr << ".";
    }
    an.makeSolution();
    auto end = std::chrono::system_clock::now();

    std::cerr << std::endl;
    const sq::VectorType<real> &E = an.get_E();
    std::cerr << "Energy : " << E.min() << std::endl;
    std::cerr << "# Steps : " << nSteps << std::endl;
    showDuration(end - start);
    std::cerr << std::endl;
}

template<class real>
void run(const char *precisionStr) {
    bool runDenseGraphBruteForceSearchers = true;
    bool runDenseGraphAnnealers = true;
    bool runBipartiteGraphBruteForceSearchers = true;
    bool runBipartiteGraphAnnealers = true;

    bool runCPUSolvers = true;
    bool runCUDASolvers = true;
    if (!sq::isCUDAAvailable())
        runCUDASolvers = false;

    /* Dense graph brute-force searchers */
    if (runDenseGraphBruteForceSearchers) {
        int N = 24;
        sq::random.seed(0);
        sq::MatrixType<real> W = symmetricMatrix<real>(N);
        if (runCPUSolvers) {
            fprintf(stderr, "Dense graph brute-force searcher, CPU, %s\n", precisionStr);
            sq::cpu::DenseGraphBFSearcher<real> searcher;
            searcher.setProblem(W);
            runSearch(searcher);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Dense graph brute-force searcher, CUDA, %s\n", precisionStr);
            sq::cuda::DenseGraphBFSearcher<real> searcher(device);
            searcher.setProblem(W);
            runSearch(searcher);
        }
#endif
    }

    /* Dense graph annealers */
    if (runDenseGraphAnnealers) {
        int N = 1024;
        sq::random.seed(0);
        sq::MatrixType<real> W = symmetricMatrix<real>(N);
        if (runCPUSolvers) {
            fprintf(stderr, "Dense graph annealer, CPU, %s\n", precisionStr);
            fprintf(stderr, "N = %d, m = %d\n", N, N / 2);
            sq::cpu::DenseGraphAnnealer<real> annealer;
            annealer.setProblem(W);
            sq::Preference pref(sq::pnNumTrotters, sq::SizeType(N / 2));
            annealer.setPreference(pref);
            anneal(annealer);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Dense graph annealer, CUDA, %s\n", precisionStr);
            fprintf(stderr, "N = %d, m = %d\n", N, N / 2);
            sq::cuda::DenseGraphAnnealer<real> annealer(device);
            annealer.setProblem(W);
            sq::Preference pref(sq::pnNumTrotters, sq::SizeType(N / 2));
            annealer.setPreference(pref);
            anneal(annealer);
        }
#endif
    }

    /* Bipartite graph brute-force searchers */
    if (runBipartiteGraphBruteForceSearchers) {
        int N0 = 14, N1 = 14;
        sq::random.seed(0);
        sq::VectorType<real> b0, b1;
        sq::MatrixType<real> W;
        createBipartiteGraph(&b0, &b1, &W, N0, N1);
        if (runCPUSolvers) {
            fprintf(stderr, "Bipartite graph brute-force searcher, CPU, %s\n", precisionStr);
            fprintf(stderr, "(N0, N1) = (%d, %d)\n", N0, N1);
            sq::cpu::BipartiteGraphBFSearcher<real> searcher;
            searcher.setProblem(b0, b1, W);
            runSearch(searcher);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Bipartite graph brute-force searcher, CUDA, %s\n", precisionStr);
            sq::cuda::BipartiteGraphBFSearcher<real> searcher(device);
            searcher.setProblem(b0, b1, W);
            runSearch(searcher);
        }
#endif
    }

    /* Bipartite graph annealers */
    if (runBipartiteGraphAnnealers) {
        int N0 = 384, N1 = 384;
        sq::random.seed(0);
        sq::VectorType<real> b0, b1;
        sq::MatrixType<real> W;
        createBipartiteGraph(&b0, &b1, &W, N0, N1);
        if (runCPUSolvers) {
            fprintf(stderr, "Bipartite graph annealer, CPU, %s\n", precisionStr);
            sq::cpu::BipartiteGraphAnnealer<real> annealer;
            annealer.setProblem(b0, b1, W);
            sq::Preference pref(sq::pnNumTrotters, sq::SizeType((N0 + N1) / 2));
            annealer.setPreference(pref);
            anneal(annealer);
        }
#ifdef SQAODC_CUDA_ENABLED
        if (runCUDASolvers) {
            fprintf(stderr, "Bipartite graph annealer, CUDA, %s\n", precisionStr);
            sq::cuda::BipartiteGraphAnnealer<real> annealer(device);
            annealer.setProblem(b0, b1, W);
            sq::Preference pref(sq::pnNumTrotters, sq::SizeType((N0 + N1) / 2));
            annealer.setPreference(pref);
            anneal(annealer);
        }
#endif
    }
}

int main() {
    bool runFloatSolvers = true;
    bool runDoubleSolvers = true;

#ifdef SQAODC_CUDA_ENABLED
    if (sq::isCUDAAvailable())
        device.initialize();
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
