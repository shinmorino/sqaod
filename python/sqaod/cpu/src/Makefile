.SUFFIXES: .cpp .o .h .c

include incpath
INCLUDE+=-I../../../../include -I../../../../libsqaod -I../../../../libsqaod/eigen

TARGETS=../cpu_formulas.so ../cpu_dg_annealer.so ../cpu_dg_bf_solver.so ../cpu_bg_annealer.so ../cpu_bg_bf_solver.so
cpu_formulas_so_OBJS=cpu_formulas.o
cpu_dg_annealer_so_OBJS=cpu_dg_annealer.o
cpu_dg_bf_solver_so_OBJS=cpu_dg_bf_solver.o
cpu_bg_annealer_so_OBJS=cpu_bg_annealer.o
cpu_bg_bf_solver_so_OBJS=cpu_bg_bf_solver.o

CXX=g++
CC=gcc
CFLAGS=-std=c++11 -fpic -DPIC -Wall -g -I$(dir $(abspath Makefile))
#CFLAGS+=-O2
CXXFLAGS=$(CFLAGS)
LDFLAGS=-L.. -lsqaod

all: $(TARGETS)

../cpu_formulas.so: $(cpu_formulas_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_dg_annealer.so: $(cpu_dg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_dg_bf_solver.so: $(cpu_dg_bf_solver_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_bg_annealer.so: $(cpu_bg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_bg_bf_solver.so: $(cpu_bg_bf_solver_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cpu_formulas_so_OBJS) $(cpu_dg_annealer_so_OBJS) $(cpu_bg_annealer_so_OBJS) $(cpu_dg_bf_solver_so_OBJS)
