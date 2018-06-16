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
        print('N={0}, {1}'.format(N, pkg.__name__))
        W = dense_graph_random(N, dtype)
        an = pkg.dense_graph_annealer(W, dtype=dtype)
        an.set_preferences(n_trotters = N)
        result = benchmark.anneal(an) 
        results.append((N, ) + result)

    return results

def run_bipartite_graph_benchmark(Nlist, dtype, pkg) :
    results = []
    for N in Nlist :
        print('N={0}, {1}'.format(N, pkg.__name__))
        b0, b1, W = bipartite_graph_random(N / 2, N / 2, dtype)
        an = pkg.bipartite_graph_annealer(b0, b1, W, dtype=dtype)
        an.set_preferences(n_trotters = N)
        result = benchmark.anneal(an) 
        results.append((N, ) + result)

    return results

if __name__ == '__main__' :
    benchmark.duration = 60.
    Nlist = [ 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3574, 4096, 5120, 6144, 7168, \
              8192 ]
    
    results = run_dense_graph_benchmark(Nlist, np.float32, sq.cpu)
    report.write('cpu_dense_graph.csv', results)

    results = run_bipartite_graph_benchmark(Nlist, np.float32, sq.cpu)
    report.write('cpu_bipartite_graph.csv', results)
    
    results = run_dense_graph_benchmark(Nlist, np.float32, sq.cuda)
    report.write('cuda_dense_graph.csv', results)

    results = run_bipartite_graph_benchmark(Nlist, np.float32, sq.cuda)
    report.write('cuda_bipartite_graph.csv', results)
