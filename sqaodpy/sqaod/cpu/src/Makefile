.SUFFIXES: .cpp .o .h .c

include incpath

TARGETS= ../cpu_dg_annealer.so ../cpu_dg_bf_searcher.so ../cpu_bg_annealer.so ../cpu_bg_bf_searcher.so ../cpu_formulas.so
cpu_formulas_so_OBJS=cpu_formulas.o
cpu_dg_annealer_so_OBJS=cpu_dg_annealer.o
cpu_dg_bf_searcher_so_OBJS=cpu_dg_bf_searcher.o
cpu_bg_annealer_so_OBJS=cpu_bg_annealer.o
cpu_bg_bf_searcher_so_OBJS=cpu_bg_bf_searcher.o

CXX=g++
CC=gcc
CFLAGS=-g -O2 -Wall -ggdb -Wno-format-security -I../../../.. -fopenmp -fPIC
CXXFLAGS=$(CFLAGS) -std=c++11
LDFLAGS=-L../../../../sqaodc/.libs -lsqaodc

all: $(TARGETS)

../cpu_formulas.so: $(cpu_formulas_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_dg_annealer.so: $(cpu_dg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_dg_bf_searcher.so: $(cpu_dg_bf_searcher_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_bg_annealer.so: $(cpu_bg_annealer_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

../cpu_bg_bf_searcher.so: $(cpu_bg_bf_searcher_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $< $(LDFLAGS)  -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cpu_formulas_so_OBJS) $(cpu_dg_annealer_so_OBJS) $(cpu_bg_annealer_so_OBJS) $(cpu_dg_bf_searcher_so_OBJS) $(cpu_bg_bf_searcher_so_OBJS)
