#pragma once

#include <sqaodc/common/internal/ParallelWorkDistributor_omp.h>

#include <sqaodc/common/internal/ParallelWorkDistributor_LockFree.h>
#include <sqaodc/common/internal/ParallelWorkDistributor_cpp.h>

#ifdef __linux__
#include <sqaodc/common/internal/ParallelWorkDistributor_linux.h>
#endif

namespace sqaod_internal {

typedef ParallelWorkDistributor_omp ParallelWorkDistributor;

// typedef ParallelWorkDistributor_cpp ParallelWorkDistributor;
// typedef ParallelWorkDistributor_LockFree ParallelWorkDistributor;

}
