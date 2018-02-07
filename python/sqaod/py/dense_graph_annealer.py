import numpy as np
import sqaod
import formulas
from types import MethodType



class DenseGraphAnnealer :

    # algorithm
    simple = 0
    stencil = 1
    
    def __init__(self, W, optimize, n_trotters) :
        self._m = None
        self._q = None
        if not W is None :
            self.set_problem(W, optimize)
            self.set_solver_preference(n_trotters)

    def _vars(self) :
        return self._h, self._J, self._c, self._q
    
    def set_problem(self, W, optimize = sqaod.minimize) :
        h, J, c = formulas.dense_graph_calculate_hJc(W)
        self._optimize = optimize
        self._h, self._J, self._c = optimize.sign(h), optimize.sign(J), optimize.sign(c)
        self._N = W.shape[0]
        
    def set_solver_preference(self, n_trotters) :
        # The default value assumed to N / 4
        m = max(2, n_trotters if n_trotters is not None else self._N / 4)
        self._m = m

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return self._E

    def get_x(self) :
        return self._x

    # TODO: adding set_x ??
    
    # Ising model

    def get_hJc(self) :
        return self._h, self._J, self._c

    def get_q(self) :
        return self._q
    
    def randomize_q(self) :
        # FIXME: raise exception
        assert self._m is not None
        if self._q is None :
            self._q = np.empty((self._m, self._N), dtype=np.int8)
        sqaod.randomize_qbits(self._q)

    def calculate_E(self) :
        h, J, c, q = self._vars()
        E = formulas.dense_graph_batch_calculate_E_from_qbits(h, J, c, q)
        self._E = self._optimize.sign(E)

    def init_anneal(self) :
        if self._m is None :
            self.set_solver_preference(None)
        if self._q is None :
            self.randomize_q()

    def fin_anneal(self) :
        self._x = []
        for idx in range(self._m) :
            x = sqaod.bits_from_qbits(self._q[idx])
            self._x.append(x)
        self.calculate_E()
            
def anneal_one_step_simple(self, G, kT) :
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

def anneal_one_step_stencil_half(self, G, kT, offset) :
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

def anneal_one_step_stencil(self, G, kT) :
    offset = np.random.randint(2)
    #offset = 0
    self.anneal_one_step_stencil_half(G, kT, offset)
    offset = (offset + 1) % 2
    self.anneal_one_step_stencil_half(G, kT, offset)

                
def dense_graph_annealer(W = None, optimize = sqaod.minimize, n_trotters = None,
                         algorithm = DenseGraphAnnealer.stencil) :
    an = DenseGraphAnnealer(W, optimize, n_trotters)
    if algorithm == DenseGraphAnnealer.simple :
        an.anneal_one_step = MethodType(anneal_one_step_simple, an, DenseGraphAnnealer)
    else :
        an.anneal_one_step = MethodType(anneal_one_step_stencil, an, DenseGraphAnnealer)
        an.anneal_one_step_stencil_half = \
                        MethodType(anneal_one_step_stencil_half, an, DenseGraphAnnealer)
    return an

if __name__ == '__main__' :


    Ginit = 5.
    Gfin = 0.01
    
    nRepeat = 4
    kT = 0.02
    tau = 0.99
    
    N = 8
    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    
    ann = dense_graph_annealer(W, sqaod.minimize, N / 2)
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.init_anneal()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.fin_anneal()
        E = ann.get_E()
        x = ann.get_x()
        print E, x

    ann = dense_graph_annealer(W, sqaod.maximize, N / 2)
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.init_anneal()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.fin_anneal()
        E = ann.get_E()
        x = ann.get_x()
        print E, x
