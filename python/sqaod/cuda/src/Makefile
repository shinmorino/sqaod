.SUFFIXES: .cpp .o .h .c

include incpath
INCLUDE+=-I../../../../libsqaod/include -I../../../../libsqaod -I../../../../libsqaod/eigen -I/usr/local/cuda-9.1/include

TARGETS= ../cuda_dg_annealer.so ../cuda_dg_bf_searcher.so ../cuda_bg_annealer.so ../cuda_bg_bf_searcher.so ../cuda_formulas.so ../cuda_device.so
cuda_formulas_so_OBJS=cuda_formulas.o
cuda_dg_annealer_so_OBJS=cuda_dg_annealer.o
cuda_dg_bf_searcher_so_OBJS=cuda_dg_bf_searcher.o
cuda_bg_annealer_so_OBJS=cuda_bg_annealer.o
cuda_bg_bf_searcher_so_OBJS=cuda_bg_bf_searcher.o
cuda_device_so_OBJS=cuda_device.o

CXX=g++
CC=gcc
CFLAGS=-std=c++11 -fpic -DPIC -Wall -ggdb -I$(dir $(abspath Makefile)) -Wno-format-security
#CFLAGS+=-O2
CXXFLAGS=$(CFLAGS)
LDFLAGS=-L.. -lsqaod_cuda -lsqaod

all: $(TARGETS)

../cuda_formulas.so: $(cuda_formulas_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_dg_annealer.so: $(cuda_dg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_dg_bf_searcher.so: $(cuda_dg_bf_searcher_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_bg_annealer.so: $(cuda_bg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_bg_bf_searcher.so: $(cuda_bg_bf_searcher_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cuda_device.so: $(cuda_device_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cuda_formulas_so_OBJS) $(cuda_dg_annealer_so_OBJS) $(cuda_bg_annealer_so_OBJS) $(cuda_dg_bf_searcher_so_OBJS) $(cuda_device_so_OBJS) $(TARGETS)
