import sqaod.py
import sqaod.cpu
import sqaod.common as common
import numpy as np


def output(solver) :
    summary = common.make_summary(solver)
    print 'E {}'.format(summary.E)
    print 'Number of solutions : {}'.format(len(summary.xlist))
    for x in summary.xlist :
        print x


def anneal(ann) :
    print ann.__class__
    
    Ginit = 5.
    Gfin = 0.01

    nRepeat = 1
    kT = 0.02
    tau = 0.99

    for loop in range(0, nRepeat) :
        common.anneal(ann, Ginit, Gfin, kT)
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

ann = sqaod.py.dense_graph_annealer(W, sqaod.minimize)
anneal(ann)
ann = sqaod.cpu.dense_graph_annealer(W)
anneal(ann)

sol = sqaod.py.dense_graph_bf_solver(W)
search(sol)
sol = sqaod.cpu.dense_graph_bf_solver(W)
search(sol)
