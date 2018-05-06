#pragma once

#include <sqaodc/common/Common.h>

namespace sqaod {

namespace cpu {

template<class real>
sqaod::DenseGraphBFSearcher<real> *newDenseGraphBFSearcher();

template<class real>
sqaod::DenseGraphAnnealer<real> *newDenseGraphAnnealer();

template<class real>
sqaod::DenseGraphFormulas<real> *newDenseGraphFormulas();

template<class real>
sqaod::BipartiteGraphBFSearcher<real> *newBipartiteGraphBFSearcher();

template<class real>
sqaod::BipartiteGraphAnnealer<real> *newBipartiteGraphAnnealer();

template<class real>
sqaod::BipartiteGraphFormulas<real> *newBipartiteGraphFormulas();

}

void deleteInstance(NullBase *);

}

#if defined(SQAODC_CUDA_ENABLED) && !defined(SQAODC_IGNORE_CUDA_HEADERS)

#include <sqaodc/cuda/api.h>

namespace sqaod {

namespace cuda {

template<class real>
DenseGraphBFSearcher<real> *newDenseGraphBFSearcher();

template<class real>
DenseGraphAnnealer<real> *newDenseGraphAnnealer();

template<class real>
DenseGraphFormulas<real> *newDenseGraphFormulas();

template<class real>
BipartiteGraphBFSearcher<real> *newBipartiteGraphBFSearcher();

template<class real>
BipartiteGraphAnnealer<real> *newBipartiteGraphAnnealer();

template<class real>
BipartiteGraphFormulas<real> *newBipartiteGraphFormulas();

Device *newDevice(int devNo = -1);

}

}

#endif
