AUTOMAKE_OPTIONS=foreign
ACLOCAL_AMFLAGS=-I m4
SUBDIRS=common cpu

libdir = ${prefix}/lib/$(build_cpu)-$(build_os)


if CUDA_ENABLED
    SUBDIRS+=cuda
endif
SUBDIRS+=tests

lib_LTLIBRARIES=libsqaodc.la
libsqaodc_la_LIBADD=$(top_builddir)/sqaodc/common/libcommon.la $(top_builddir)/sqaodc/cpu/libcpu.la -ldl
libsqaodc_la_SOURCES=
libsqaodc_la_LDFLAGS=-version-info 0:0:0

if WITH_BLAS
    libsqaodc_la_LIBADD+=-lblas
endif


if CUDA_ENABLED
    lib_LTLIBRARIES+=libsqaodc_cuda.la
    libsqaodc_cuda_la_SOURCES=
    libsqaodc_cuda_la_LIBADD=cuda/libcuda.la libsqaodc.la -L@CUDA_LIBDIR@ -lcublas -lcudart -lcurand
    libsqaodc_cuda_la_LDFLAGS=-version-info 0:0:0
endif

# see how to set version info
# https://www.glamenv-septzen.net/view/484
