import numpy as np
import random
import sqaod
import sqaod.common as common
import sqaod.py as py
import cpu_dg_annealer as dg_annealer

class DenseGraphAnnealer :
    
    def __init__(self, W, optimize, n_trotters, dtype) :
        self.dtype = dtype
        self._ext = dg_annealer.new_annealer(dtype)
        self.set_problem(W, optimize)
        self.set_solver_preference(n_trotters)

    def __del__(self) :
        dg_annealer.delete_annealer(self._ext, self.dtype)
        
    def rand_seed(seed) :
        dg_annealer.rand_seed(self._ext, seed)
        
    def set_problem(self, W, optimize = sqaod.minimize) :
        # FIXME: check W dim
        self._N = W.shape[0]
        W = common.clone_as_np_buffer(W, self.dtype)
        dg_annealer.set_problem(self._ext, W, optimize, self.dtype)

    def set_solver_preference(self, n_trotters = None) :
        if n_trotters is None :
            n_trotters = self._N / 4
        self._m = n_trotters;
        self._E = np.zeros((self._m), self.dtype)
        dg_annealer.set_solver_preference(self._ext, n_trotters, self.dtype)

    def randomize_q(self) :
        dg_annealer.randomize_q(self._ext, self.dtype)

    def get_q(self) :
        q = np.empty((self._m, self._N), np.int8)
        dg_annealer.get_q(self._ext, q, self.dtype)
        return q

    def get_hJc(self) :
        h = np.empty((self._N), self.dtype)
        J = np.empty((self._N, self._N), self.dtype)
        c = np.empty((1), self.dtype)
        dg_annealer.get_hJc(self._ext, h, J, c, self.dtype)
        return h, J, c[0]

    def get_E(self) :
        dg_annealer.get_E(self._ext, self._E, self.dtype)
        return np.min(self._E);

    def calculate_E(self) :
        dg_annealer.calculate_E(self._ext, self.dtype)

    def init_anneal(self) :
        if not hasattr(self, '_m') :
            self.set_solver_preference(None)

    def anneal_one_step(self, G, kT) :
        dg_annealer.anneal_one_step(self._ext, G, kT, self.dtype)
        

def dense_graph_annealer(W, optimize=sqaod.minimize, n_trotters = None, dtype=np.float64) :
    return DenseGraphAnnealer(W, optimize, n_trotters, dtype)


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
    W = common.generate_random_symmetric_W(N, -0.5, 0.5, np.float64)

    ann = dense_graph_annealer(W, n_trotters = m, dtype=np.float64)
#    ann = py.dense_graph_annealer(N, m)
    ann.set_problem(W, sqaod.minimize)

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
        ann.init_anneal()
        ann.randomize_q()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.calculate_E()
        E = ann.get_E()
        q = ann.get_q() 
        print E
        print q
