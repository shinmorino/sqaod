import numpy as np
import sqaod
import formulas
from sqaod.common import checkers
from types import MethodType
from sqaod import algorithm as algo

class DenseGraphAnnealer :

    def __init__(self, W, optimize, prefdict) :
        if not W is None :
            self.set_problem(W, optimize)
        self._select_algorithm(algo.coloring)
        self.set_preferences(prefdict)

    def seed(self, seed) :
        # py version uses using global random generator.
        # Nothing to do here.
        pass
        
    def _vars(self) :
        return self._h, self._J, self._c, self._q
    
    def get_problem_size(self) :
        return self.N_;

    def set_problem(self, W, optimize = sqaod.minimize) :
        checkers.dense_graph.qubo(W)

        h, J, c = formulas.dense_graph_calculate_hJc(W)
        self._optimize = optimize
        self._h, self._J, self._c = optimize.sign(h), optimize.sign(J), optimize.sign(c)
        self._N = W.shape[0]
        self._m = self._N / 2
        
    def _select_algorithm(self, algoname) :
        if algoname == algo.naive :
            self.anneal_one_step = \
                MethodType(DenseGraphAnnealer.anneal_one_step_naive, self)
        else :
            self.anneal_one_step = \
                MethodType(DenseGraphAnnealer.anneal_one_step_coloring, self)

    def _get_algorithm(self) :
        if self.anneal_one_step is self.anneal_one_step_naive :
            return algo.naive;
        return algo.coloring
            
    def set_preferences(self, prefdict = None, **prefs) :
        if not prefdict is None :
            self._set_prefdict(prefdict)
        self._set_prefdict(prefs)
        
    def _set_prefdict(self, prefdict) :
        v = prefdict.get('n_trotters')
        if v is not None :
            self._m = v;
        v = prefdict.get('algorithm')
        if v is not None :
            self._select_algorithm(v)

    def get_preferences(self) :
        prefs = { }
        if hasattr(self, '_m') :
            prefs['n_trotters'] = self._m
        prefs['algorithm'] = self._get_algorithm()
        return prefs
        
    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return self._E

    def get_x(self) :
        return self._x

    def set_x(self, x) :
        self._x[:] = x
    
    # Ising model

    def get_hJc(self) :
        return self._h, self._J, self._c

    def get_q(self) :
        return self._q
    
    def randomize_spin(self) :
        sqaod.randomize_spin(self._q)

    def calculate_E(self) :
        h, J, c, q = self._vars()
        E = formulas.dense_graph_batch_calculate_E_from_spin(h, J, c, q)
        self._E = self._optimize.sign(E)

    def init_anneal(self) :
        self._q = np.empty((self._m, self._N), dtype=np.int8)

    def fin_anneal(self) :
        self._x = []
        for idx in range(self._m) :
            x = sqaod.bit_from_spin(self._q[idx])
            self._x.append(x)
        self.calculate_E()

    def anneal_one_step(self, G, kT) :
        # will be dynamically replaced.
        pass
        
    def anneal_one_step_naive(self, G, kT) :
        h, J, c, q = self._vars()
        N = self._N
        m = self._m
        two_div_m = 2. / np.float64(m)
        coef = np.log(np.tanh(G/kT/m)) / kT
        
        for i in range(self._N * self._m):
            x = np.random.randint(N)
            y = np.random.randint(m)
            qyx = q[y][x]
            sum = np.dot(J[x], q[y]); # diagnoal elements in J are zero.
            dE = - two_div_m * qyx * (h[x] + sum)
            dE -= qyx * (q[(m + y - 1) % m][x] + q[(y + 1) % m][x]) * coef
            threshold = 1. if (dE <= 0.) else np.exp(-dE / kT)
            if threshold > np.random.rand():
                q[y][x] = - qyx

    def anneal_colored_plane(self, G, kT, offset) :
        h, J, c, q = self._vars()
        N = self._N
        m = self._m
        two_div_m = 2. / np.float64(m)
        coef = np.log(np.tanh(G/kT/m)) / kT
        
        for y in range(self._m):
            x = (offset + np.random.randint(1 << 30) * 2) % N
            qyx = q[y][x]
            sum = np.dot(J[x], q[y]); # diagnoal elements in J are zero.
            dE = - two_div_m * qyx * (h[x] + sum)
            dE -= qyx * (q[(m + y - 1) % m][x] + q[(y + 1) % m][x]) * coef
            threshold = 1. if (dE <= 0.) else np.exp(-dE / kT)
            if threshold > np.random.rand():
                q[y][x] = - qyx
            
    def anneal_one_step_coloring(self, G, kT) :
        for loop in range(0, self._N) :
            self.anneal_colored_plane(G, kT, 0)
            self.anneal_colored_plane(G, kT, 1)

                
def dense_graph_annealer(W = None, optimize = sqaod.minimize, **prefs) :
    an = DenseGraphAnnealer(W, optimize, prefs)
    return an


if __name__ == '__main__' :

    np.random.seed(0)
    Ginit = 5.
    Gfin = 0.01
    
    nRepeat = 4
    kT = 0.02
    tau = 0.99
    
    N = 8
    m = 4
    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    

    #N = 20
    #m = 10
    #W = sqaod.generate_random_symmetric_W(N, -0.5, 0.5, np.float64)
    
    algoname = algo.default
    # algo = DenseGraphAnnealer.naive
    ann = dense_graph_annealer(W, sqaod.minimize, n_trotters=m)
    ann.set_preferences(algorithm = algo.naive)
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.init_anneal()
        ann.randomize_spin()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.fin_anneal()
        E = ann.get_E()
        #x = ann.get_x()
        print E #, x

    prefs = ann.get_preferences()
    print prefs
        
    ann = dense_graph_annealer(W, sqaod.maximize)
    ann.set_preferences(prefs)
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.init_anneal()
        ann.randomize_spin()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.fin_anneal()
        E = ann.get_E()
        #x = ann.get_x()
        print E #, x

    prefs = ann.get_preferences()
    print prefs
