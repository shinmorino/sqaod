#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using BFSearcher = sq::cpu::DenseGraphBFSearcher<real>;

}

#define modname "cpu_dg_bf_searcher"
#define INIT_MODULE INITFUNCNAME(cpu_dg_bf_searcher)
#define DENSE_GRAPH

#include <sqaodc/pyglue/bf_searcher.inc>
