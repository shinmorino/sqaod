#pragma once

#include <sqaodc/common/Common.h>

/* cpu */
#include <sqaodc/cpu/CPUDenseGraphBFSearcher.h>
#include <sqaodc/cpu/CPUDenseGraphAnnealer.h>
#include <sqaodc/cpu/CPUBipartiteGraphBFSearcher.h>
#include <sqaodc/cpu/CPUBipartiteGraphAnnealer.h>
#include <sqaodc/cpu/CPUFormulas.h>

#if defined(SQAODC_CUDA_ENABLED) && !defined(SQAODC_IGNORE_CUDA_HEADERS)
/* cuda */
#include <sqaodc/cuda/CUDADenseGraphBFSearcher.h>
#include <sqaodc/cuda/CUDADenseGraphAnnealer.h>
#include <sqaodc/cuda/CUDABipartiteGraphBFSearcher.h>
#include <sqaodc/cuda/CUDABipartiteGraphAnnealer.h>
#include <sqaodc/cuda/CUDAFormulas.h>
#include <sqaodc/cuda/Device.h>
#endif

namespace sqaod {

namespace native {

namespace cpu {

template<class real>
using DenseGraphBFSearcher = sqaod_cpu::CPUDenseGraphBFSearcher<real>;

template<class real>
using DenseGraphAnnealer = sqaod_cpu::CPUDenseGraphAnnealer<real>;

template<class real>
using DenseGraphFormulas = sqaod_cpu::CPUDenseGraphFormulas<real>;

template<class real>
using BipartiteGraphBFSearcher = sqaod_cpu::CPUBipartiteGraphBFSearcher<real>;

template<class real>
using BipartiteGraphAnnealer = sqaod_cpu::CPUBipartiteGraphAnnealer<real>;

template<class real>
using BipartiteGraphFormulas = sqaod_cpu::CPUBipartiteGraphFormulas<real>;

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

}

}
