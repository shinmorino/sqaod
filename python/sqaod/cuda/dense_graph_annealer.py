import numpy as np
import random
import sqaod
from sqaod.common import checkers
import cuda_dg_annealer as dg_annealer
import device

class DenseGraphAnnealer :
    
    def __init__(self, W, optimize, n_trotters, dtype) :
        self.dtype = dtype
        self._ext = dg_annealer.new_annealer(dtype)
	self.assign_device(device.active_device)
	self._device = device.active_device;
        if not W is None :
            self.set_problem(W, optimize)
            self.set_preferences(n_trotters = W.shape[0] / 4)

    def __del__(self) :
        dg_annealer.delete_annealer(self._ext, self.dtype)
        
    def assign_device(self, dev) :
        dg_annealer.assign_device(self._ext, dev._ext, self.dtype)

    def seed(self, seed) :
        dg_annealer.seed(self._ext, seed, self.dtype)
        
    def set_problem(self, W, optimize = sqaod.minimize) :
        checkers.dense_graph.qubo(W)
        W = sqaod.clone_as_ndarray(W, self.dtype)
        dg_annealer.set_problem(self._ext, W, optimize, self.dtype)
        self._optimize = optimize

    def get_problem_size(self) :
        return dg_annealer.get_problem_size(self._ext, self.dtype)

    def set_preferences(self, **prefs) :
        N = self.get_problem_size()
        dg_annealer.set_preferences(self._ext, prefs, self.dtype)

    def get_preferences(self) :
	return dg_annealer.get_preferences(self._ext, self.dtype)

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return self._E

    def get_x(self) :
        return dg_annealer.get_x(self._ext, self.dtype)

    def get_hJc(self) :
        N = self.get_problem_size()
        h = np.empty((N), self.dtype)
        J = np.empty((N, N), self.dtype)
        c = np.empty((1), self.dtype)
        dg_annealer.get_hJc(self._ext, h, J, c, self.dtype)
        return h, J, c[0]

    def get_q(self) :
        return dg_annealer.get_q(self._ext, self.dtype)

    def randomize_q(self) :
        dg_annealer.randomize_q(self._ext, self.dtype)

    def calculate_E(self) :
        dg_annealer.calculate_E(self._ext, self.dtype)

    def init_anneal(self) :
        dg_annealer.init_anneal(self._ext, self.dtype)

    def fin_anneal(self) :
        dg_annealer.fin_anneal(self._ext, self.dtype)
        N = self.get_problem_size()
	prefs = self.get_preferences()
	m = prefs['n_trotters']
        self._E = np.empty((m), self.dtype)
        dg_annealer.get_E(self._ext, self._E, self.dtype)

    def anneal_one_step(self, G, kT) :
        dg_annealer.anneal_one_step(self._ext, G, kT, self.dtype)
        

def dense_graph_annealer(W = None, optimize=sqaod.minimize, n_trotters = None, dtype=np.float64) :
    ann = DenseGraphAnnealer(W, optimize, n_trotters, dtype)
    return ann


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

    N = 100
    m = 5
    W = sqaod.generate_random_symmetric_W(N, -0.5, 0.5, np.float64)

    ann = dense_graph_annealer(W, n_trotters = m, dtype=np.float64)
    import sqaod.py as py
    #ann = py.dense_graph_annealer(W, n_trotters = m)
    ann.set_problem(W, sqaod.minimize)
    h, J, c = ann.get_hJc()
    print h
    print J
    print c

    
    Ginit = 5.
    Gfin = 0.001
    
    nRepeat = 2
    kT = 0.02
    tau = 0.995
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.init_anneal()
        ann.randomize_q()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau
        ann.fin_anneal()
        ann.calculate_E()
        E = ann.get_E()
        q = ann.get_q() 
        print E
        print q
