#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using BFSearcher = sq::cpu::BipartiteGraphBFSearcher<real>;

}

#define modname "cpu_bg_bf_searcher"
#define INIT_MODULE INITFUNCNAME(cpu_bg_bf_searcher)
#define BIPARTITE_GRAPH

#include <sqaodc/pyglue/bf_searcher.inc>
