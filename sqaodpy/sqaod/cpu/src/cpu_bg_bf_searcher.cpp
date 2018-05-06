#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using BFSearcher = sq::BipartiteGraphBFSearcher<real>;

template<class real>
BFSearcher<real> *newBFSearcher() {
    return sqaod::cpu::newBipartiteGraphBFSearcher<real>();
}

}

#define modname "cpu_bg_bf_searcher"
#define INIT_MODULE INITFUNCNAME(cpu_bg_bf_searcher)
#define BIPARTITE_GRAPH

#include <sqaodc/pyglue/bf_searcher.inc>
