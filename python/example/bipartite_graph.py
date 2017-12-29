import sqaod.py
import sqaod.cpu
import sqaod.common as common
import numpy as np


def anneal(ann) :
    
    Ginit = 5.
    Gfin = 0.01

    nRepeat = 1
    kT = 0.02
    tau = 0.99

    for loop in range(0, nRepeat) :
        common.anneal(ann, Ginit, Gfin, kT)
        x = ann.get_x() 
        E = ann.get_E()

        print x
        print E


def search(sol) :
    sol.search()
    x = sol.get_x()
    E = sol.get_E()
    print x
    print E
    
        
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
