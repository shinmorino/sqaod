check_PROGRAMS=test perf

LDADD=
LIBS=


test_SOURCES=main.cpp MinimalTestSuite.cpp CUDAFormulasBGFuncTest.cpp CUDAFormulasDGFuncTest.cpp DeviceRandomTest.cpp CUDADenseGraphBFSolverTest.cpp DeviceSegmentedSumTest.cpp CUDADenseGraphAnnealerTest.cpp
perf_SOURCES=perf.cpp


if CUDA_ENABLED
  test_SOURCES+=DeviceMathTest.cpp DeviceTest.cpp
  LDADD+=$(top_builddir)/cuda/libcuda.la
  LIBS+=-L/usr/local/cuda/lib64 -lcublas -lcudart
endif
if WITH_BLAS
  LIBS+=-lblas
endif

LDADD+=$(top_builddir)/cpu/libcpu.la $(top_builddir)/common/libcommon.la

run: $(check_PROGRAMS)
	./test
