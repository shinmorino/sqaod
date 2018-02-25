import numpy as np
import random
import sqaod
from sqaod.common import checkers
import cuda_dg_annealer as cext
import device

class DenseGraphAnnealer :

    _active_device = device.active_device
    _cext = cext
    
    def __init__(self, W, optimize, dtype, prefdict) :
        self.dtype = dtype
        self._cobj = cext.new_annealer(dtype)
	self.assign_device(device.active_device)
        if not W is None :
            self.set_problem(W, optimize)
        self.set_preferences(prefdict)

    def __del__(self) :
        cext.delete_annealer(self._cobj, self.dtype)
        
    def assign_device(self, device) :
        cext.assign_device(self._cobj, device._cobj, self.dtype)

    def seed(self, seed) :
        cext.seed(self._cobj, seed, self.dtype)
        
    def set_problem(self, W, optimize = sqaod.minimize) :
        checkers.dense_graph.qubo(W)
        W = sqaod.clone_as_ndarray(W, self.dtype)
        cext.set_problem(self._cobj, W, optimize, self.dtype)
        self._optimize = optimize

    def get_problem_size(self) :
        return cext.get_problem_size(self._cobj, self.dtype)

    def set_preferences(self, prefdict = None, **prefs) :
        if not prefdict is None:
            cext.set_preferences(self._cobj, prefdict, self.dtype)
        cext.set_preferences(self._cobj, prefs, self.dtype)

    def get_preferences(self) :
	return cext.get_preferences(self._cobj, self.dtype)

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return self._E

    def get_x(self) :
        return cext.get_x(self._cobj, self.dtype)

    def get_hJc(self) :
        N = self.get_problem_size()
        h = np.empty((N), self.dtype)
        J = np.empty((N, N), self.dtype)
        c = np.empty((1), self.dtype)
        cext.get_hJc(self._cobj, h, J, c, self.dtype)
        return h, J, c[0]

    def get_q(self) :
        return cext.get_q(self._cobj, self.dtype)

    def randomize_q(self) :
        cext.randomize_q(self._cobj, self.dtype)

    def calculate_E(self) :
        cext.calculate_E(self._cobj, self.dtype)

    def init_anneal(self) :
        cext.init_anneal(self._cobj, self.dtype)

    def fin_anneal(self) :
        cext.fin_anneal(self._cobj, self.dtype)
        N = self.get_problem_size()
	prefs = self.get_preferences()
	m = prefs['n_trotters']
        self._E = np.empty((m), self.dtype)
        cext.get_E(self._cobj, self._E, self.dtype)

    def anneal_one_step(self, G, kT) :
        cext.anneal_one_step(self._cobj, G, kT, self.dtype)
        

def dense_graph_annealer(W = None, optimize=sqaod.minimize, dtype=np.float64, **prefs) :
    ann = DenseGraphAnnealer(W, optimize, dtype, prefs)
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

    ann = dense_graph_annealer(W, dtype=np.float64, n_trotters = m)
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
