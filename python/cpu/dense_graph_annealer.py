import numpy as np
import random
import solver_traits
import utils
import py
from py import tags
import cpu_dg_annealer as dg_annealer

class DenseGraphAnnealer :
    
    def __init__(self, N, m, dtype) :
        self.dtype = dtype
        self.ext = dg_annealer.new_annealer(dtype)
        self.set_problem_size(N, m)

    def rand_seed(seed) :
        dg_annealer.rand_seed(self.ext, seed)
        
    def set_problem_size(self, N, m) :
        self.N = N
        self.m = m;
        self.E = np.zeros((self.m), self.dtype)
        dg_annealer.set_problem_size(self.ext, N, m, self.dtype)
        
    def set_problem(self, W, optimize = tags.minimize) :
        W = solver_traits.clone_as_np_buffer(W, self.dtype)
        dg_annealer.set_problem(self.ext, W, optimize, self.dtype)

    def randomize_q(self) :
        dg_annealer.randomize_q(self.ext, self.dtype)

    def get_q(self) :
        q = np.empty((self.m, self.N), np.int8)
        dg_annealer.get_q(self.ext, q, self.dtype)
        return q

    def get_hJc(self) :
        h = np.empty((self.N), self.dtype)
        J = np.empty((self.N, self.N), self.dtype)
        c = np.empty((1), self.dtype)
        dg_annealer.get_hJc(self.ext, h, J, c, self.dtype)
        return h, J, c[0]

    def get_E(self) :
        dg_annealer.get_E(self.ext, self.E, self.dtype)
        return self.E;

    def calculate_E(self) :
        dg_annealer.calculate_E(self.ext, self.dtype)

    def anneal_one_step(self, G, kT) :
        dg_annealer.anneal_one_step(self.ext, G, kT, self.dtype)
        

def dense_graph_annealer(N = 0, m = 0, dtype=np.float64) :
    return DenseGraphAnnealer(N, m, dtype)


if __name__ == '__main__' :

    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    np.random.seed(0)

    N = 8
    m = 4
    """
    N = 2
    m = 2
    W = np.array([[-32,4],
                  [4,-32]])
    """
    
    N = 200
    m = 150
    W = utils.generate_random_symmetric_W(N, -0.5, 0.5, np.float64)

    ann = dense_graph_annealer(N, m, dtype=np.float64)
#    ann = py.dense_graph_annealer(N, m)
    ann.set_problem(W, tags.minimize)

    h, J, c = ann.get_hJc()
    print h
    print J
    print c
    
    
    Ginit = 5.
    Gfin = 0.001
    
    nRepeat = 4
    kT = 0.02
    tau = 0.995
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.randomize_q()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.calculate_E()
        q = ann.get_q() 
        E = ann.get_E()
        print(q, E)
