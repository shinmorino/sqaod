AUTOMAKE_OPTIONS=foreign
ACLOCAL_AMFLAGS=-I m4
SUBDIRS=common cpu

nobase_pkginclude_HEADERS=sqaodc.h common/sqaod_config.h common/defines.h common/types.h common/Common.h common/Random.h common/Matrix.h common/Array.h common/Solver.h common/Preference.h common/Formulas.h common/os_dependent.h cuda/api.h pyglue/pyglue.h pyglue/annealer.inc pyglue/bf_searcher.inc pyglue/formulas.inc
libdir = @libdir@/$(build_cpu)-$(build_os)


if CUDA_ENABLED
    SUBDIRS+=cuda
endif
SUBDIRS+=tests

lib_LTLIBRARIES=libsqaodc.la
libsqaodc_la_LIBADD=$(top_builddir)/sqaodc/common/libcommon.la $(top_builddir)/sqaodc/cpu/libcpu.la
libsqaodc_la_SOURCES=
if WITH_BLAS
    libsqaodc_la_LIBADD+=-lblas
endif
libsqaodc_la_LIBADD+=-ldl


if CUDA_ENABLED
    lib_LTLIBRARIES+=libsqaodc_cuda.la
    libsqaodc_cuda_la_SOURCES=
    libsqaodc_cuda_la_LIBADD=cuda/libcuda.la libsqaodc.la
    libsqaodc_cuda_la_LDFLAGS=-version-info 0:0:0 @CUDA_LDFLAGS@ -lcublas -lcurand -lcudart
endif

# see how to set version info
# https://www.glamenv-septzen.net/view/484
