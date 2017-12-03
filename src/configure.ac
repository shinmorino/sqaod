# AC_PREREQ([2.69])
AC_INIT(foo, 1.0, joe@foo.com)    
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE
AC_PROG_LIBTOOL
AC_CONFIG_SRCDIR([config.h.in])

# Using the former instead of the latter AC_CONFIG_HEADERS results in
# automake complaining about lack of confdefs.h.in.
# autoheader doesn't help
#AC_CONFIG_HEADERS([config.h] [confdefs.h])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])

AC_PROG_CC
AC_PROG_CXX
#if test x$prefix = xNONE ; then
#  prefix=${ac_default_prefix}
#fi

# I want this stuff to go into config.h or confdefs.h
# AC_DEFINE([STUFF], ["blahblah"], [Stuff])
# AC_DEFINE_UNQUOTED(DOCDIR, ["${prefix}/share/doc/foo"], [Documentation])

CFLAGS="-O2 -Wall -ggdb"
CXXFLAGS=$CFLAGS

AC_SUBST([CFLAGS], $CFLAGS)
AC_SUBST([CXXFLAGS], $CFLAGS)


AC_CONFIG_FILES([Makefile cpu/Makefile])
AC_OUTPUT
