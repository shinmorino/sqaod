import sqaod.py
import sqaod.cpu
import sqaod.common as common
import numpy as np


def anneal(ann) :
    
    Ginit = 5.
    Gfin = 0.01

    nRepeat = 4
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
    
        

W = np.array([[-32,4,4,4,4,4,4,4],
              [4,-32,4,4,4,4,4,4],
              [4,4,-32,4,4,4,4,4],
              [4,4,4,-32,4,4,4,4],
              [4,4,4,4,-32,4,4,4],
              [4,4,4,4,4,-32,4,4],
              [4,4,4,4,4,4,-32,4],
              [4,4,4,4,4,4,4,-32]])

ann = sqaod.py.dense_graph_annealer(W)
anneal(ann)
ann = sqaod.cpu.dense_graph_annealer(W)
anneal(ann)

sol = sqaod.py.dense_graph_bf_solver(W)
search(sol)

sol = sqaod.cpu.dense_graph_bf_solver(W)
search(sol)
