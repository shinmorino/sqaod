from __future__ import print_function
import sqaod
import numpy as np

if sqaod.is_cuda_available() :
    import sqaod.cuda
else :
    print 'cuda not available'


def output(searcher) :
    summary = sqaod.make_summary(searcher)
    print('E {}'.format(summary.E))
    print('Number of solutions : {}'.format(len(summary.xlist)))
    nToShow = min(len(summary.xlist), 4)
    for idx in range(nToShow) :
        print(summary.xlist[idx])


def anneal(ann) :
    print(ann.__class__)
    
    Ginit = 5.
    Gfin = 0.01

    nRepeat = 1
    beta = 1. / 0.02
    tau = 0.99

    for loop in range(0, nRepeat) :
        sqaod.anneal(ann, Ginit, Gfin, beta)
        output(ann)


def search(sol) :
    sol.search()
    output(sol)
        

W = np.array([[-32,4,4,4,4,4,4,4],
              [4,-32,4,4,4,4,4,4],
              [4,4,-32,4,4,4,4,4],
              [4,4,4,-32,4,4,4,4],
              [4,4,4,4,-32,4,4,4],
              [4,4,4,4,4,-32,4,4],
              [4,4,4,4,4,4,-32,4],
              [4,4,4,4,4,4,4,-32]])

ann = sqaod.py.dense_graph_annealer(W, sqaod.minimize, n_troggers = 4)
anneal(ann)
ann = sqaod.cpu.dense_graph_annealer(W, sqaod.minimize, n_trotters = 4)
anneal(ann)

if sqaod.is_cuda_available() :
    ann = sqaod.cuda.dense_graph_annealer(W, sqaod.minimize, n_trotters = 4)
    anneal(ann)

sol = sqaod.py.dense_graph_bf_searcher(W)
search(sol)
sol = sqaod.cpu.dense_graph_bf_searcher(W, tile_size = 4)
search(sol)

if sqaod.is_cuda_available() :
    sol = sqaod.cuda.dense_graph_bf_searcher(W, tile_size = 4)
    search(sol)
