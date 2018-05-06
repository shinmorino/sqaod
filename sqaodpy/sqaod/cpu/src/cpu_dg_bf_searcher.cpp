#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using BFSearcher = sq::DenseGraphBFSearcher<real>;

template<class real>
BFSearcher<real> *newBFSearcher() {
    return sqaod::cpu::newDenseGraphBFSearcher<real>();
}

}

#define modname "cpu_dg_bf_searcher"
#define INIT_MODULE INITFUNCNAME(cpu_dg_bf_searcher)
#define DENSE_GRAPH

#include <sqaodc/pyglue/bf_searcher.inc>
