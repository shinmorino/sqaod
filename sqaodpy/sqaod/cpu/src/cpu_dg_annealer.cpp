#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using Annealer = sq::cpu::DenseGraphAnnealer<real>;

}

#define modname "cpu_dg_annealer"
#define INIT_MODULE INITFUNCNAME(cpu_dg_annealer)

#define DENSE_GRAPH

#include <sqaodc/pyglue/annealer.inc>
