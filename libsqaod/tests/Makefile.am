check_PROGRAMS=test

AM_CPPFLAGS=-I/usr/local/cuda/include -I$(abs_top_srcdir)/eigen -I$(abs_top_srcdir)/cub  # FIXME: cofnigrue cuda path.

test_SOURCES=main.cpp MinimalTestSuite.cpp CUDAFormulasBGFuncTest.cpp CUDAFormulasDGFuncTest.cpp DeviceRandomTest.cpp CUDADenseGraphBFSolverTest.cpp
test_LDADD=
LIBS=

if CUDA_ENABLED
  test_SOURCES+=DeviceMathTest.cpp DeviceTest.cpp
  test_LDADD+=$(top_builddir)/cuda/libcuda.la
	LIBS+=-L/usr/local/cuda/lib64 -lcublas -lcudart
endif

test_LDADD+=$(top_builddir)/cpu/libcpu.la $(top_builddir)/common/libcommon.la

run: $(check_PROGRAMS)
	./test

