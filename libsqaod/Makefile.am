AUTOMAKE_OPTIONS=foreign
ACLOCAL_AMFLAGS=-I m4
SUBDIRS=common cpu

if CUDA_ENABLED
    SUBDIRS+=cuda
endif
SUBDIRS+=tests

lib_LTLIBRARIES=libsqaod.la
libsqaod_la_LIBADD=$(top_builddir)/common/libcommon.la $(top_builddir)/cpu/libcpu.la
libsqaod_la_SOURCES=

if WITH_BLAS
    libsqaod_la_LIBADD+=-lblas
endif


if CUDA_ENABLED
    lib_LTLIBRARIES+=libsqaod_cuda.la
    libsqaod_cuda_la_SOURCES=
    libsqaod_cuda_la_LIBADD=cuda/libcuda.la libsqaod.la -L@CUDA_LIBDIR@ -lsqaod -lcublas -lcudart -lcurand
endif

