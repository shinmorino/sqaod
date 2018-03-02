#pragma once

#include <sqaodc/common/Common.h>
#include <sqaodc/cpu/cpu_export.h>

#if defined(SQAODC_CUDA_ENABLED) && !defined(SQAODC_IGNORE_CUDA_HEADERS)
#include <sqaodc/cuda/cuda_export.h>
#endif

namespace sqaod {

namespace cpu {

template<class real>
using DenseGraphBFSearcher = sqaod_cpu::CPUDenseGraphBFSearcher<real>;

template<class real>
using DenseGraphAnnealer = sqaod_cpu::CPUDenseGraphAnnealer<real>;

template<class real>
using DenseGraphFormulas = sqaod_cpu::DGFuncs<real>;

template<class real>
using BipartiteGraphBFSearcher = sqaod_cpu::CPUBipartiteGraphBFSearcher<real>;

template<class real>
using BipartiteGraphAnnealer = sqaod_cpu::CPUBipartiteGraphAnnealer<real>;

template<class real>
using BipartiteGraphFormulas = sqaod_cpu::BGFuncs<real>;

}

#if defined(SQAODC_CUDA_ENABLED) && !defined(SQAODC_IGNORE_CUDA_HEADERS)

namespace cuda {

template<class real>
using DenseGraphBFSearcher = sqaod_cuda::CUDADenseGraphBFSearcher<real>;

template<class real>
using DenseGraphAnnealer = sqaod_cuda::CUDADenseGraphAnnealer<real>;

template<class real>
using DenseGraphFormulas = sqaod_cuda::CUDADenseGraphFormulas<real>;


template<class real>
using BipartiteGraphBFSearcher = sqaod_cuda::CUDABipartiteGraphBFSearcher<real>;

template<class real>
using BipartiteGraphAnnealer = sqaod_cuda::CUDABipartiteGraphAnnealer<real>;

template<class real>
using BipartiteGraphFormulas = sqaod_cuda::CUDABipartiteGraphFormulas<real>;

typedef sqaod_cuda::Device Device;

}

#endif

};
