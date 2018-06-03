from __future__ import print_function
import sqaod as sq
import numpy as np
import benchmark
import report

def dense_graph_random(N, dtype) :
    return sq.generate_random_symmetric_W(N, dtype=dtype)

def bipartite_graph_random(N0, N1, dtype = np.float64) :
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    W = np.random.random((N1, N0)) - 0.5
    return b0, b1, W

def run_dense_graph_benchmark(Nlist, dtype, pkg) :
    results = []
    for N in Nlist :
        W = dense_graph_random(N, dtype)
        searcher = pkg.dense_graph_bf_searcher(W, dtype=dtype)
        result = benchmark.search(searcher) 
        results.append((N, ) + result)

    return results

def run_bipartite_graph_benchmark(Nlist, dtype, pkg) :
    results = []
    for N in Nlist :
        b0, b1, W = bipartite_graph_random(N / 2, N / 2, dtype)
        searcher = pkg.bipartite_graph_bf_searcher(b0, b1, W, dtype=dtype)
        result = benchmark.search(searcher) 
        results.append((N, ) + result)

    return results

if __name__ == '__main__' :
    # Nlist = [ 8, 16, 20, 24, 28, 32, 36 ]
    benchmark.duration = 10.
    Nlist = [ 8, 16, 20 ]
    
    results = run_dense_graph_benchmark(Nlist, np.float32, sq.cpu)
    report.write('dense_graph_bf_searcher.csv', results)

    results = run_bipartite_graph_benchmark(Nlist, np.float32, sq.cpu)
    report.write('bipartite_graph_bf_searcher.csv', results)
