#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using Annealer = sq::cpu::BipartiteGraphAnnealer<real>;

}

#define modname "cpu_bg_annealer"
#define INIT_MODULE INITFUNCNAME(cpu_bg_annealer)
#define BIPARTITE_GRAPH

#include <sqaodc/pyglue/annealer.inc>
