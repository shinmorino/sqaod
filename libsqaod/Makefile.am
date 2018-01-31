AUTOMAKE_OPTIONS=foreign
ACLOCAL_AMFLAGS=-I m4
SUBDIRS=common cpu

if CUDA_ENABLED
    SUBDIRS+=cuda
endif

SUBDIRS+=tests


lib_LTLIBRARIES=libsqaod.la

libsqaod_la_SOURCES=
libsqaod_la_LIBADD=$(top_builddir)/common/libcommon.la $(top_builddir)/cpu/libcpu.la
if WITH_BLAS
    libsqaod_la_LIBADD+=-lblas
endif

LIBS=
if CUDA_ENABLED
    LIBS+=-L@CUDA_LIBDIR@ -lcublas -lcudart -lcurand
endif

