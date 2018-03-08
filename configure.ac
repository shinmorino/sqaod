# AC_PREREQ([2.69])
AC_INIT(sqaodc, 0.1.0, shin.morino@gmail.com)    
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([configure.ac])

LT_INIT([disable-static])


# Using the former instead of the latter AC_CONFIG_HEADERS results in
# automake complaining about lack of confdefs.h.in.
# autoheader doesn't help
#AC_CONFIG_HEADERS([config.h] [confdefs.h])
AC_CONFIG_HEADERS([config/config.h])
AC_CONFIG_MACRO_DIRS([m4])


# CUDA macros
# https://tschoonj.github.io/blog/2014/05/10/building-a-cuda-device-function-library-with-autotools/
# https://www.gnu.org/software/automake/manual/html_node/Usage-of-Conditionals.html

AC_ARG_ENABLE([cuda],
[  --enable-cuda    enable CUDA solvers. [[default=no]]],
[case "${enableval}" in
  yes) cuda=true; cuda_prefix="/usr/local/cuda" ;;
  no)  cuda=false ;;
  *) cuda=true; cuda_prefix=$enableval;;
esac],[cuda=false])
AM_CONDITIONAL([CUDA_ENABLED], [test x$cuda = xtrue])

if test z${CUDA_ARCH} = z
then
  CUDA_ARCH="35 50 60 61 70"
fi

for arch in ${CUDA_ARCH}
  do
    NVCCFLAGS="$NVCCFLAGS -gencode arch=compute_$arch,code=sm_$arch"
done
NVCCFLAGS="$NVCCFLAGS -gencode arch=compute_70,code=compute_70"
echo "NVCC options : $NVCCFLAGS"

AC_ARG_WITH([blas],
[  --with-blas    use CPU blas.  [[default=yes]]],
[case "${withval}" in
  yes) blas=true ;;
  *)  blas=false ;;
esac],[blas=true])
AM_CONDITIONAL([WITH_BLAS], [test x${blas} = xtrue])

blasdef='#define SQAODC_WITH_BLAS' 
if test x$blas = xfalse
then
  blasdef="/* ${blasdef} */"
fi
AC_SUBST([DEFINE_SQAODC_WITH_BLAS], ${blasdef})

cudadef='#define SQAODC_CUDA_ENABLED' 
if test x$cuda = xfalse
then
  cudadef="/* ${cudadef} */"
fi
AC_SUBST([DEFINE_SQAODC_CUDA_ENABLED], ${cudadef})
AC_SUBST([NVCC], ${cuda_prefix}/bin/nvcc)
AC_SUBST([NVCCFLAGS], $NVCCFLAGS)
AC_SUBST([CUDA_LIBDIR], ${cuda_prefix}/lib64)
AC_SUBST([CUDA_INCLUDE], ${cuda_prefix}/include)
AC_PROG_CC
AC_PROG_CXX

if test x$prefix = xNONE ; then
  prefix=${ac_default_prefix}
fi

# I want this stuff to go into config.h or confdefs.h
# AC_DEFINE([STUFF], ["blahblah"], [Stuff])
# AC_DEFINE_UNQUOTED(DOCDIR, ["${prefix}/share/doc/foo"], [Documentation])

CFLAGS="$CFLAGS -Wall -ggdb -Wno-format-security -I\${abs_top_srcdir} -fopenmp"
AC_SUBST([CFLAGS], $CFLAGS)

CXXFLAGS="-std=c++11 $CFLAGS -I\${abs_top_srcdir}/eigen"
AM_COND_IF([CUDA_ENABLED],
	[CXXFLAGS="$CXXFLAGS -I${cuda_prefix}/include -I\${abs_top_srcdir}/cub"])
AC_SUBST([CXXFLAGS], $CXXFLAGS)


SQAODC_CXXFLAGS="${CXXFLAGS}"
AC_SUBST([SQAODC_CXXFLAGS], ${SQAODC_CXXFLAGS})

abs_top_builddir=
SQAODC_LDFLAGS="-L${prefix}/lib -l${abs_top_builddir}/.libs"
if test x$cuda = xtrue
then
    SQAODC_LDFLAGS="$SQAODC_LDFLAGS -lsqaod_cuda"
fi
SQAODC_LDFLAGS="$SQAODC_LDFLAGS -lsqaod"
AC_SUBST([SQAODC_LDFLAGS], $SQAODC_LDFLAGS)


AC_CONFIG_FILES([Makefile
                 sqaodc/Makefile
		 config/sqaod_config.h
		 sqaodc/common/Makefile
		 sqaodc/cpu/Makefile
		 sqaodc/cuda/Makefile
		 sqaodc/tests/Makefile])
AC_OUTPUT
