# AC_PREREQ([2.69])
AC_INIT(libsqaod, 0.0.dev0, shin.morino@gmail.com)    
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([config.h.in])

LT_INIT([disable-static])


# Using the former instead of the latter AC_CONFIG_HEADERS results in
# automake complaining about lack of confdefs.h.in.
# autoheader doesn't help
#AC_CONFIG_HEADERS([config.h] [confdefs.h])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])


# CUDA macros
# https://tschoonj.github.io/blog/2014/05/10/building-a-cuda-device-function-library-with-autotools/
# https://www.gnu.org/software/automake/manual/html_node/Usage-of-Conditionals.html

AC_ARG_ENABLE([cuda],
[  --enable-cuda    enable CUDA solvers],
[case "${enableval}" in
  yes) cuda=true ;;
  no)  cuda=false ;;
  *) cuda_prefix="/usr/local/cuda";;
esac],[cuda=false])
AM_CONDITIONAL([CUDA_ENABLED], [test x$cuda = xtrue])

AC_PROG_CC
AC_PROG_CXX
#if test x$prefix = xNONE ; then
#  prefix=${ac_default_prefix}
#fi

# I want this stuff to go into config.h or confdefs.h
# AC_DEFINE([STUFF], ["blahblah"], [Stuff])
# AC_DEFINE_UNQUOTED(DOCDIR, ["${prefix}/share/doc/foo"], [Documentation])

#CFLAGS="-std=c++11 -O2 -Wall -ggdb"
CFLAGS="-std=c++11 -Wall -ggdb"
CXXFLAGS=$CFLAGS

AC_SUBST([CFLAGS], $CFLAGS)
AC_SUBST([CXXFLAGS], $CFLAGS)

AC_CONFIG_FILES([Makefile common/Makefile cpu/Makefile])
AM_COND_IF([CUDA_ENABLED],
           [AC_CONFIG_FILES([cuda/Makefile])])
AC_CONFIG_FILES([tests/Makefile])
AC_OUTPUT
