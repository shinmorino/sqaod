#include <sqaodc/sqaodc_native.h>
#include <cuda_runtime_api.h>


extern "C"
void sqaodc_cuda_version(int *version, int *cuda_version) {
    *version = SQAODC_VERSION;
    *cuda_version = CUDART_VERSION;
}

namespace sqaod {

namespace cuda {

template<class real>
sqaod::cuda::DenseGraphBFSearcher<real> *newDenseGraphBFSearcher() {
    return new sqaod::native::cuda::DenseGraphBFSearcher<real>();
}

template<class real>
sqaod::cuda::DenseGraphAnnealer<real> *newDenseGraphAnnealer() {
    return new sqaod::native::cuda::DenseGraphAnnealer<real>();
}

template<class real>
sqaod::cuda::DenseGraphFormulas<real> *newDenseGraphFormulas() {
    return new sqaod::native::cuda::DenseGraphFormulas<real>();
}

template<class real>
sqaod::cuda::BipartiteGraphBFSearcher<real> *newBipartiteGraphBFSearcher() {
    return new sqaod::native::cuda::BipartiteGraphBFSearcher<real>();
}

template<class real>
sqaod::cuda::BipartiteGraphAnnealer<real> *newBipartiteGraphAnnealer() {
    return new sqaod::native::cuda::BipartiteGraphAnnealer<real>();
}

template<class real>
sqaod::cuda::BipartiteGraphFormulas<real> *newBipartiteGraphFormulas() {
    return new sqaod::native::cuda::BipartiteGraphFormulas<real>();
}


template sqaod::cuda::DenseGraphBFSearcher<double> *newDenseGraphBFSearcher();
template sqaod::cuda::DenseGraphBFSearcher<float> *newDenseGraphBFSearcher();

template sqaod::cuda::DenseGraphAnnealer<double> *newDenseGraphAnnealer();
template sqaod::cuda::DenseGraphAnnealer<float> *newDenseGraphAnnealer();

template sqaod::cuda::DenseGraphFormulas<double> *newDenseGraphFormulas<double>();
template sqaod::cuda::DenseGraphFormulas<float> *newDenseGraphFormulas<float>();

template sqaod::cuda::BipartiteGraphBFSearcher<double> *newBipartiteGraphBFSearcher();
template sqaod::cuda::BipartiteGraphBFSearcher<float> *newBipartiteGraphBFSearcher();

template sqaod::cuda::BipartiteGraphAnnealer<double> *newBipartiteGraphAnnealer();
template sqaod::cuda::BipartiteGraphAnnealer<float> *newBipartiteGraphAnnealer();

template sqaod::cuda::BipartiteGraphFormulas<double> *newBipartiteGraphFormulas();
template sqaod::cuda::BipartiteGraphFormulas<float> *newBipartiteGraphFormulas();


Device *newDevice(int devNo) {
    return new sqaod::native::cuda::Device(devNo);
}

}

}
