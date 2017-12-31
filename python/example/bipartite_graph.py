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
    
        
np.random.seed(0)

N0 = 10
N1 = 10

W = np.random.random((N1, N0)) - 0.5
b0 = np.random.random((N0)) - 0.5
b1 = np.random.random((N1)) - 0.5
    
ann = sqaod.py.bipartite_graph_annealer(b0, b1, W)
anneal(ann)
ann = sqaod.cpu.bipartite_graph_annealer(b0, b1, W)
anneal(ann)

sol = sqaod.py.bipartite_graph_bf_solver(b0, b1, W)
search(sol)
sol = sqaod.cpu.bipartite_graph_bf_solver(b0, b1, W)
search(sol)
