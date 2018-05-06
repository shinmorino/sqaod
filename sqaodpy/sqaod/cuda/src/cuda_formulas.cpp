#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real> using DGFormulas = sq::cuda::DenseGraphFormulas<real>;
template<class real> using BGFormulas = sq::cuda::BipartiteGraphFormulas<real>;

template<class real>
DGFormulas<real> *newDGFormulas() {
    return sqaod::cuda::newDenseGraphFormulas<real>();
}

template<class real>
BGFormulas<real> *newBGFormulas() {
    return sqaod::cuda::newBipartiteGraphFormulas<real>();
}

}

#define modname "cuda_formulas"
#define INIT_MODULE INITFUNCNAME(cuda_formulas)

#define CUDA_FORMULAS
#include <sqaodc/pyglue/formulas.inc>
