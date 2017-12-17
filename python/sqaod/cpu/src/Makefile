.SUFFIXES: .cpp .o .h .c

include incpath
INCLUDE+=-I../../../include -I../../../src -I../../../src/eigen

TARGETS=../cpu_native.so ../cpu_dg_annealer.so ../cpu_dg_bf_solver.so
cpu_native_so_OBJS=cpu_native.o
cpu_dg_annealer_so_OBJS=cpu_dg_annealer.o
cpu_dg_bf_solver_so_OBJS=cpu_dg_bf_solver.o

CXX=g++
CC=gcc
CFLAGS=-fpic -DPIC -Wall -g -I$(dir $(abspath Makefile))
#CFLAGS+=-O2
CXXFLAGS=$(CFLAGS)
LDFLAGS=-L.. -lquant_digital

all: $(TARGETS)

../cpu_native.so: $(cpu_native_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_dg_annealer.so: $(cpu_dg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_dg_bf_solver.so: $(cpu_dg_bf_solver_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cpu_native_so_OBJS) $(cpu_dg_annealer_so_OBJS) $(cpu_dg_bf_solver_so_OBJS)
