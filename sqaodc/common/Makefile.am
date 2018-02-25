noinst_LTLIBRARIES=libcommon.la
libcommon_la_SOURCES=sqaod_config.h defines.cpp Matrix.cpp Common.cpp UniformOp.cpp Random.cpp Preference.cpp Solver.cpp

AM_CPPFLAGS=-I$(top_srcdir)/..

defines.cpp: sqaod_config.h
sqaod_config.h: $(top_srcdir)/config/sqaod_config.h
	@if ! diff $@ $<  >/dev/null 2>&1 ; then cp $< $@ ; fi
