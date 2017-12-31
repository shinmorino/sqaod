import sqaod
import numpy as np


def output(solver) :
    summary = sqaod.make_summary(solver)
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
        sqaod.anneal(ann, Ginit, Gfin, kT)
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
