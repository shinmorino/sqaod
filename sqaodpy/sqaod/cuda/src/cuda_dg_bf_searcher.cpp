#include <sqaodc/pyglue/pyglue.h>
#include <sqaodc/sqaodc.h>

namespace {

template<class real>
using BFSearcher = sq::cuda::DenseGraphBFSearcher<real>;

template<class real>
BFSearcher<real> *newBFSearcher() {
    return sqaod::cuda::newDenseGraphBFSearcher<real>();
}

}

#define modname "cuda_dg_bf_searcher"
#define INIT_MODULE INITFUNCNAME(cuda_dg_bf_searcher)
#define DENSE_GRAPH
#define CUDA_SOLVER

#include <sqaodc/pyglue/bf_searcher.inc>
