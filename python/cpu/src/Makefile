.SUFFIXES: .cpp .o .h .c

include incpath
INCLUDE+=-I../../../include -I../../../src -I../../../src/eigen

TARGETS=../cpu_native.so
cpu_native_so_OBJS=cpu_native.o

CXX=g++
CC=gcc
CFLAGS=-fpic -Wall -g -I$(dir $(abspath Makefile))
#CFLAGS+=-O2
CXXFLAGS=$(CFLAGS)
LDFLAGS=-L.. -lquant_digital

all: $(TARGETS)

../cpu_native.so: $(cpu_native_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $(cpu_native_so_OBJS) $(LDFLAGS)  -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cpu_native_so_OBJS)
