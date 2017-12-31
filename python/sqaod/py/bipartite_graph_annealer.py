import numpy as np
import sqaod
import formulas
from sqaod.common import checkers


class BipartiteGraphAnnealer :

    def __init__(self, b0, b1, W, optimize, n_trotters) : # n_trotters
        self._m = None
        self._q0 = self._q1 = None
        if not W is None :
            self.set_problem(b0, b1, W, optimize)
            # FIXME
            self.set_solver_preference(n_trotters)

    def _vars(self) :
        return self._h0, self._h1, self._J, self._c, self._q0, self._q1

    def _get_dim(self) :
        return self._dim[0], self._dim[1], self._m
        
    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)

        self._dim = (W.shape[1], W.shape[0])
        self._optimize = optimize
        h0, h1, J, c = formulas.bipartite_graph_calculate_hJc(b0, b1, W)
        self._h0, self._h1 = optimize.sign(h0), optimize.sign(h1)
        self._J, self._c = optimize.sign(J), optimize.sign(c)

    def set_solver_preference(self, n_trotters) :
        # set n_trotters.  The default value assumed to N / 4
        if n_trotters is None :
            n_trotters = (self._dim[0] + self._dim[1]) / 4
        self._m = max(2, n_trotters)

    def get_optimize_dir(self) :
        return self._optimize
        
    def get_E(self) :
        return self._E
    
    def get_x(self) :
        return self._x_pairs

    def set_x(self, x0, x1) :
        if (x0.shape[len(x0.shape) - 1] != self._dim[0]) \
           or (x1.shape[len(x1.shape) - 1] != self._dim[1]) :
            raise Exception("Dim does not match.")

        if self._q0 is None :
            self._q0 = np.empty((self._m, self._dim[0]), dtype=np.int8)
        if self._q1 is None :
            self._q1 = np.empty((self._m, self._dim[1]), dtype=np.int8)
        
        q0 = sqaod.bits_to_qbits(x0)
        q1 = sqaod.bits_to_qbits(x1)
        self._q0[:][...] = q0[:]
        self._q1[:][...] = q1[:]

    # Ising model / spins
    
    def get_hJc(self) :
        return self._h, self._J, self._c
            
    def get_q(self) :
        return self._q0, self._q1

    def randomize_q(self) :
        if self._q0 is None :
            self._q0 = np.empty((self._m, self._dim[0]), dtype=np.int8)
        sqaod.randomize_qbits(self._q0)
        if self._q1 is None :
            self._q1 = np.empty((self._m, self._dim[1]), dtype=np.int8)
        sqaod.randomize_qbits(self._q1)

    def init_anneal(self) :
        if self._m is None :
            self.set_solver_preference(None)
        self.randomize_q()

    def fin_anneal(self) :
        self._x_pairs = []
        for idx in range(self._m) :
            x0 = sqaod.bits_from_qbits(self._q0[idx])
            x1 = sqaod.bits_from_qbits(self._q1[idx])
            self._x_pairs.append((x0, x1))
        self.calculate_E()
        
    def _anneal_half_step(self, N, qAnneal, h, J, qFixed, G, kT, m) :
        dEmat = np.matmul(J, qFixed.T)
        twoDivM = 2. / m
        tempCoef = np.log(np.tanh(G/kT/m)) / kT
        invKT = 1. / kT
        for loop in range(N * m) :
            iq = np.random.randint(N)
            im = np.random.randint(m)
            q = qAnneal[im][iq]
            dE = - twoDivM * q * (h[iq] + dEmat[iq, im])
            mNeibour0 = (im + m - 1) % m
            mNeibour1 = (im + 1) % m
            dE -= q * (qAnneal[mNeibour0][iq] + qAnneal[mNeibour1][iq]) * tempCoef
            thresh = 1 if dE < 0 else np.exp(- dE * invKT) 
            if thresh > np.random.rand():
                qAnneal[im][iq] = -q
                    
    def anneal_one_step(self, G, kT) :
        h0, h1, J, c, q0, q1 = self._vars()
        N0, N1, m = self._get_dim()
        self._anneal_half_step(N1, q1, h1, J, q0, G, kT, m)
        self._anneal_half_step(N0, q0, h0, J.T, q1, G, kT, m)

    def calculate_E(self) :
        h0, h1, J, c, q0, q1 = self._vars()
        E = np.empty((self._m), J.dtype)
        for idx in range(self._m) :
            # FIXME: 1d output for batch calculation
            E[idx] = formulas.bipartite_graph_calculate_E_from_qbits(h0, h1, J, c, q0[idx], q1[idx])
        self._E = self._optimize.sign(E)

def bipartite_graph_annealer(b0 = None, b1 = None, W = None, \
                             optimize = sqaod.minimize, n_trotters = None) :
    return BipartiteGraphAnnealer(b0, b1, W, optimize, n_trotters)


if __name__ == '__main__' :
    N0 = 10
    N1 = 10
    m = 10
    
    np.random.seed(0)
            
    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5

    an = bipartite_graph_annealer(b0, b1, W, sqaod.minimize, m)
    
    Ginit = 5
    Gfin = 0.01
    kT = 0.02
    tau = 0.99
    n_repeat = 10

    for loop in range(0, n_repeat) :
        an.init_anneal()
        G = Ginit
        while Gfin < G :
            an.anneal_one_step(G, kT)
            G = G * tau
        an.fin_anneal()
        
        E = an.get_E()
        x = an.get_x()
        print E
        print x
