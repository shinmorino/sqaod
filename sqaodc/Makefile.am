AUTOMAKE_OPTIONS=foreign
ACLOCAL_AMFLAGS=-I m4
SUBDIRS=common cpu

if CUDA_ENABLED
    SUBDIRS+=cuda
endif
SUBDIRS+=tests

lib_LTLIBRARIES=libsqaodc.la
libsqaodc_la_LIBADD=$(top_builddir)/common/libcommon.la $(top_builddir)/cpu/libcpu.la -ldl
libsqaodc_la_SOURCES=

if WITH_BLAS
    libsqaodc_la_LIBADD+=-lblas
endif


if CUDA_ENABLED
    lib_LTLIBRARIES+=libsqaodc_cuda.la
    libsqaodc_cuda_la_SOURCES=
    libsqaodc_cuda_la_LIBADD=cuda/libcuda.la libsqaodc.la -L@CUDA_LIBDIR@ -lcublas -lcudart -lcurand
endif

