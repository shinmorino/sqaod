from __future__ import print_function
import sqaod as sq
import numpy as np
import timeit
import sys

duration = 60.

def anneal(an) :
    G = 0.01
    beta = 1 / 0.02
    
    an.prepare()
    an.randomize_spin()
    timer = timeit.default_timer
    warmup = 0.
    nIters = 3
    while warmup < 5. :
        begin = timer();
        for i in range(nIters) :
            # print('.', end='', file=sys.stderr)
            an.anneal_one_step(G, beta);
        an.make_solution()
        end = timer()
        warmup = end - begin
        # print(warmup)
        if warmup < 5. :
            nIters *= 3
            print('nIters(updated) = {0}'.format(nIters))
            
    elapsedTimePerIter = warmup / nIters        
    nIters = int(duration / warmup * nIters) + 1
    print('Done.')
    print('Averarge time : {0} sec, Expected # iterations in {1} sec. : {2}'
          .format(elapsedTimePerIter, duration, nIters))
    sys.stdout.flush()
    
    begin = timer();
    for i in range(nIters) :
        # print('.', end='', file=sys.stderr)
        an.anneal_one_step(G, beta);
    an.make_solution()
    end = timer()
    elapsedTime = end - begin
    elapsedTimePerIter = elapsedTime / nIters
    print('Done,')
    print('Averarge time : {0} sec, {1} iterations executed in {2} sec.'
          .format(elapsedTimePerIter, nIters, elapsedTime))
    sys.stdout.flush()
    
    return nIters, elapsedTimePerIter


def search(searcher) :
    searcher.prepare()
    timer = timeit.default_timer
    warmup = 0.
    nIters = 1
    while warmup < 5. :
        begin = timer();
        for i in range(nIters) :
            # print('.', end='', file=sys.stderr)
            searcher.search();
        searcher.make_solution()
        end = timer()
        warmup = end - begin
        # print(warmup)
        if warmup < 5. :
            nIters *= 3
            print('nIters(updated) = {0}'.format(nIters))
            
    elapsedTimePerIter = warmup / nIters        
    nIters = int(duration / warmup * nIters) + 1
    print('Done.')
    print('Averarge time : {0} sec, Expected # iterations in {1} sec. : {2}'
          .format(elapsedTimePerIter, duration, nIters))
    sys.stdout.flush()
    
    begin = timer();
    for i in range(nIters) :
        # print('.', end='', file=sys.stderr)
        searcher.search();
    searcher.make_solution()
    end = timer()
    elapsedTime = end - begin
    elapsedTimePerIter = elapsedTime / nIters
    print('Done,')
    print('Averarge time : {0} sec, {1} iterations executed in {2} sec.'
          .format(elapsedTimePerIter, nIters, elapsedTime))
    sys.stdout.flush()
    
    return nIters, elapsedTimePerIter

if __name__ == '__main__' :
    N = 2048
    W = sq.generate_random_symmetric_W(N, dtype=dtype)
    W = dense_graph_random(N, np.float32)
    an = sq.cpu.dense_graph_annealer(W, dtype=np.float32)
    an.set_preferences(n_trotters = N)
    elapsed, nIters = benchmark(an)
    print(elapsed, nIters)
