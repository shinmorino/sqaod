#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using DGFormulas = sq::cpu::DenseGraphFormulas<real>;

template<class real>
using BGFormulas = sq::cpu::BipartiteGraphFormulas<real>;

}

#define modname "cpu_formulas"
#define INIT_MODULE INITFUNCNAME(cpu_formulas)

#include <sqaodc/pyglue/formulas.inc>
