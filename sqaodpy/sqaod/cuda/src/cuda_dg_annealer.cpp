#include <sqaodc/pyglue/pyglue.h>

namespace {

template<class real>
using Annealer = sq::cuda::DenseGraphAnnealer<real>;

}

#define modname "cuda_dg_annealer"
#define INIT_MODULE INITFUNCNAME(cuda_dg_annealer)
#define DENSE_GRAPH
#define CUDA_SOLVER

#include <sqaodc/pyglue/annealer.inc>
