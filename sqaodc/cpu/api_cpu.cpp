#include <sqaodc/sqaodc_native.h>

namespace sqaod {

namespace cpu {

template<class real>
sqaod::DenseGraphBFSearcher<real> *newDenseGraphBFSearcher() {
    return new sqaod_cpu::CPUDenseGraphBFSearcher<real>();
}

template<class real>
sqaod::DenseGraphAnnealer<real> *newDenseGraphAnnealer() {
    return new sqaod_cpu::CPUDenseGraphAnnealer<real>();
}

template<class real>
sqaod::DenseGraphFormulas<real> *newDenseGraphFormulas() {
    return new sqaod_cpu::CPUDenseGraphFormulas<real>();
}

template<class real>
sqaod::BipartiteGraphBFSearcher<real> *newBipartiteGraphBFSearcher() {
    return new sqaod_cpu::CPUBipartiteGraphBFSearcher<real>();
}

template<class real>
sqaod::BipartiteGraphAnnealer<real> *newBipartiteGraphAnnealer() {
    return new sqaod_cpu::CPUBipartiteGraphAnnealer<real>();
}

template<class real>
sqaod::BipartiteGraphFormulas<real> *newBipartiteGraphFormulas() {
    return new sqaod_cpu::CPUBipartiteGraphFormulas<real>();
}


template sqaod::DenseGraphBFSearcher<double> *newDenseGraphBFSearcher();
template sqaod::DenseGraphBFSearcher<float> *newDenseGraphBFSearcher();

template sqaod::DenseGraphAnnealer<double> *newDenseGraphAnnealer();
template sqaod::DenseGraphAnnealer<float> *newDenseGraphAnnealer();

template sqaod::DenseGraphFormulas<double> *newDenseGraphFormulas<double>();
template sqaod::DenseGraphFormulas<float> *newDenseGraphFormulas<float>();

template sqaod::BipartiteGraphBFSearcher<double> *newBipartiteGraphBFSearcher();
template sqaod::BipartiteGraphBFSearcher<float> *newBipartiteGraphBFSearcher();

template sqaod::BipartiteGraphAnnealer<double> *newBipartiteGraphAnnealer();
template sqaod::BipartiteGraphAnnealer<float> *newBipartiteGraphAnnealer();

template sqaod::BipartiteGraphFormulas<double> *newBipartiteGraphFormulas();
template sqaod::BipartiteGraphFormulas<float> *newBipartiteGraphFormulas();

}

}
