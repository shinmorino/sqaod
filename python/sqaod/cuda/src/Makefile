.SUFFIXES: .cpp .o .h .c

include incpath
INCLUDE+=-I../../../../libsqaod/include -I../../../../libsqaod -I../../../../libsqaod/eigen -I/usr/local/cuda-9.1/include

TARGETS= ../cuda_dg_annealer.so ../cuda_dg_bf_solver.so ../cuda_bg_annealer.so ../cuda_bg_bf_solver.so ../cuda_formulas.so
cuda_formulas_so_OBJS=cuda_formulas.o
cuda_dg_annealer_so_OBJS=cuda_dg_annealer.o
cuda_dg_bf_solver_so_OBJS=cuda_dg_bf_solver.o
cuda_bg_annealer_so_OBJS=cuda_bg_annealer.o
cuda_bg_bf_solver_so_OBJS=cuda_bg_bf_solver.o

CXX=g++
CC=gcc
CFLAGS=-std=c++11 -fpic -DPIC -Wall -ggdb -I$(dir $(abspath Makefile)) -Wno-format-security
#CFLAGS+=-O2
CXXFLAGS=$(CFLAGS)
LDFLAGS=-L.. -lsqaod

all: $(TARGETS)

../cuda_formulas.so: $(cuda_formulas_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_dg_annealer.so: $(cuda_dg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_dg_bf_solver.so: $(cuda_dg_bf_solver_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_bg_annealer.so: $(cuda_bg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_bg_bf_solver.so: $(cuda_bg_bf_solver_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cuda_formulas_so_OBJS) $(cuda_dg_annealer_so_OBJS) $(cuda_bg_annealer_so_OBJS) $(cuda_dg_bf_solver_so_OBJS)
