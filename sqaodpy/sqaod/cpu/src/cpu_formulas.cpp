#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real> using DGFormulas = sq::DenseGraphFormulas<real>;
template<class real> using BGFormulas = sq::BipartiteGraphFormulas<real>;

template<class real>
DGFormulas<real> *newDGFormulas() {
    return sqaod::cpu::newDenseGraphFormulas<real>();
}

template<class real>
BGFormulas<real> *newBGFormulas() {
    return sqaod::cpu::newBipartiteGraphFormulas<real>();
}

}

#define modname "cpu_formulas"
#define INIT_MODULE INITFUNCNAME(cpu_formulas)

#include <sqaodc/pyglue/formulas.inc>
