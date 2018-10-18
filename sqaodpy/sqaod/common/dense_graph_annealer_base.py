from __future__ import print_function
import numpy as np
from . import checkers
from . import preference as pref
from . import common

class DenseGraphAnnealerBase :
    
    def __init__(self, cext, dtype, W, optimize, prefdict) :
        self.dtype = dtype
        self._cext = cext
        if not W is None :
            self.set_qubo(W, optimize)
        self.set_preferences(prefdict)

    def __del__(self) :
        if hasattr(self, '_cobj') :
            self._cext.delete(self._cobj, self.dtype)
        
    def seed(self, seed) :
        self._cext.seed(self._cobj, seed, self.dtype)
        
    def set_qubo(self, W, optimize = pref.minimize) :
        checkers.dense_graph.qubo(W)
        W = common.fix_type(W, self.dtype)
        W = common.symmetrize(W)
        
        self._cext.set_qubo(self._cobj, W, optimize, self.dtype)
        self._optimize = optimize

    def get_problem_size(self) :
        return self._cext.get_problem_size(self._cobj, self.dtype)

    def set_preferences(self, prefdict = None, **prefs) :
        if not prefdict is None:
            self._cext.set_preferences(self._cobj, prefdict, self.dtype)
        self._cext.set_preferences(self._cobj, prefs, self.dtype)

    def get_preferences(self) :
        return self._cext.get_preferences(self._cobj, self.dtype)

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return self._cext.get_E(self._cobj, self.dtype)

    def calculate_E(self) :
        self._cext.calculate_E(self._cobj, self.dtype)

    def get_x(self) :
        return self._cext.get_x(self._cobj, self.dtype)

    def set_q(self, q) :
        q = common.fix_type(q, np.int8)
        self._cext.set_q(self._cobj, q, self.dtype)

    def set_qset(self, qset) :
        qlist = []
        for qvec in qset :
            qvec = common.fix_type(qvec, np.int8)
            qlist.append(qvec)
        self._cext.set_qset(self._cobj, qlist, self.dtype)

    def get_hamiltonian(self) :
        N = self.get_problem_size()
        h = np.empty((N), self.dtype)
        J = np.empty((N, N), self.dtype)
        c = np.empty((1), self.dtype)
        self._cext.get_hamiltonian(self._cobj, h, J, c, self.dtype)
        return h, J, c[0]

    def set_hamiltonian(self, h, J, c) :
        checkers.dense_graph.hJc(h, J, c)
        
        h, J = common.fix_type([h, J], self.dtype)
        J = common.symmetrize(J)
        
        c = self.dtype(c)
        self._cext.set_hamiltonian(self._cobj, h, J, c, self.dtype)
        self._optimize = pref.minimize

    def get_q(self) :
        return self._cext.get_q(self._cobj, self.dtype)

    def randomize_spin(self) :
        self._cext.randomize_spin(self._cobj, self.dtype)

    def prepare(self) :
        self._cext.prepare(self._cobj, self.dtype)

    def make_solution(self) :
        self._cext.make_solution(self._cobj, self.dtype)

    def anneal_one_step(self, G, beta) :
        G, beta = self.dtype(G), self.dtype(beta)
        self._cext.anneal_one_step(self._cobj, G, beta, self.dtype)
