import numpy as np
import random
import solver_traits
import utils
import tags

class DenseGraphAnnealer :
    
    def __init__(self, W, optimize, n_trotters) :
        if not W is None :
            self.set_problem(W, optimize)
            self.set_solver_prefereence(n_trotters)

    def _vars(self) :
        return self.h, self.J, self.c, self.q

    def _Esign(self) :
        return self._optimize.Esign
    
    def set_problem(self, W, optimize = tags.minimize) :
        self.h, self.J, self.c = solver_traits.dense_graph_calculate_hJc(W)
        self._optimize = optimize
        Esign = self._Esign()
        self.h, self.J, self.c = Esign * self.h, Esign * self.J, Esign * self.c
        self._N = N

    def set_solver_preference(self, n_trotters) :
        # The default value assumed to N / 4
        m = max(2, n_trotters if n_trotters is not None else self._N / 4)
        self._m = m
        self.q = np.zeros((self._m, self._N), dtype=np.int8)

    def get_E(self) :
        Emin = np.min(self._E)
        return self._Esign() * Emin

    def get_solutions(self) :
        Esign = self._Esign()
        sols = []
        for idx in range(self._m) :
            x = utils.bits_from_qbits(self.q[idx])
            sols.append((Esign * self._E[idx], x))
        return sols

    # Ising model
    
    def randomize_q(self) :
        utils.randomize_qbits(self.q)

    def get_hJc(self) :
        return self.h, self.J, self.c

    def calculate_E(self) :
        h, J, c, q = self._vars()
        self._E = solver_traits.dense_graph_batch_calculate_E_from_qbits(h, J, c, q)

    def init_anneal(self) :
        if not hasattr(self, '_m') :
            self.set_solver_preference(self._N / 4)
            self.randomize_q()
        
    def anneal_one_step(self, G, kT) :
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
            if np.exp(-dE / kT) > np.random.rand():
                q[y][x] = - qyx
                
def dense_graph_annealer(W = None, optimize=tags.minimize, n_trotters = None) :
    return DenseGraphAnnealer(W, optimize, n_trotters)


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
    
    ann = dense_graph_annealer(W, tags.minimize, N / 2)
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.randomize_q()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.calculate_E()
        E = ann.get_E()
        q = ann.get_solutions()
        print E, q

    ann = dense_graph_annealer(W, tags.maximize, N / 2)
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.randomize_q()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.calculate_E()
        E = ann.E()
        q = ann.get_solutions()
        print E, q
