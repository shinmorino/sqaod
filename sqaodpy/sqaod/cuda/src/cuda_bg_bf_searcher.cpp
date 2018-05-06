#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using BFSearcher = sq::cuda::BipartiteGraphBFSearcher<real>;

template<class real>
BFSearcher<real> *newBFSearcher() {
    return sqaod::cuda::newBipartiteGraphBFSearcher<real>();
}

}

#define modname "cuda_bg_bf_searcher"
#define INIT_MODULE INITFUNCNAME(cuda_bg_bf_searcher)
#define BIPARTITE_GRAPH
#define CUDA_SOLVER

#include <sqaodc/pyglue/bf_searcher.inc>
