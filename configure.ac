# AC_PREREQ([2.69])
AC_INIT(sqaodc, 0.1.1, shin.morino@gmail.com)    
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
  yes) cuda=true; cuda_prefix="default" ;;
  no)  cuda=false; cuda_prefix="na" ;;
  *) cuda=true; cuda_prefix=$enableval;;
esac],[cuda=false])

# override CUDA prefix setting if CUDA_PREFIX env var is given.
if test z${CUDA_PREFIX} != z
then
  cuda_prefix=${CUDA_PREFIX}
  cuda=true
fi

AM_CONDITIONAL([CUDA_ENABLED], [test x$cuda = xtrue])

if test $cuda = true
then
  if test z${cuda_prefix} = zdefault
  then
    cuda_prefix=`ls -d /usr/local/cuda-* | tail -1`
  fi
  echo "CUDA Prefix : $cuda_prefix"

  cuda_ver=`echo $cuda_prefix | grep -o -e 9\.[[0-9]]`
  echo "CUDA Version : $cuda_ver"

  if test $cuda_ver = $cuda_prefix
  then
    cuda_prefix=/usr/local/cuda-$cuda_ver
  fi

  cuda_package_ver=`echo $cuda_ver | tr . -`

  AC_SUBST([CUDA_VER], $cuda_ver)
  AC_SUBST([CUDA_PACKAGE_VER], $cuda_package_ver)

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

fi

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

AM_COND_IF([CUDA_ENABLED],
	[AC_SUBST([NVCC], ${cuda_prefix}/bin/nvcc)
	 AC_SUBST([NVCCFLAGS], $NVCCFLAGS)
	 AC_SUBST([CUDA_PREFIX], "${cuda_prefix}")
	 AC_SUBST([CUDA_LDFLAGS], "-L${cuda_prefix}/lib64")
	 AC_SUBST([CUDA_INCLUDE], "-I${cuda_prefix}/include")])
AC_PROG_CC
AC_PROG_CXX

if test x$prefix = xNONE ; then
  prefix=${ac_default_prefix}
fi

case z${SIMD_OPT} in
     zavx2) CFLAGS="$CFLAGS -mavx2 -mfma" ;;
     *)    CFLAGS="$CFLAGS -m${SIMD_OPT}" ;;
esac
AC_SUBST([SIMD_POSTFIX], "-${SIMD_OPT}")

CFLAGS="$CFLAGS -Wall -ggdb -Wno-format-security -I\${abs_top_srcdir} -fopenmp"
AC_SUBST([CFLAGS], $CFLAGS)

CXXFLAGS="-std=c++11 $CFLAGS -I\${abs_top_srcdir}/3rdparty/eigen"
AM_COND_IF([CUDA_ENABLED],
	[CXXFLAGS="$CXXFLAGS -I${cuda_prefix}/include -I\${abs_top_srcdir}/3rdparty/cub"])
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
		 sqaodc/tests/Makefile
		 sqaodpy/sqaod/cuda/src/Makefile
		 sqaodpy/sqaod/cpu/src/Makefile
])
AC_OUTPUT

top_builddir=.
top_srcdir=`dirname $0`
export src=${top_builddir}/config/sqaod_config.h
export dst=${top_srcdir}/sqaodc/common/sqaod_config.h
if ! diff $src $dst  >/dev/null 2>&1
then
  cp $src $dst
  echo sqaod_config.h updated.
fi
