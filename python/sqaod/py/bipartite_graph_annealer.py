import numpy as np
import random
import sqaod
import sqaod.common as common
import formulas


class BipartiteGraphAnnealer :

    def __init__(self, W, b0, b1, optimize, n_trotters) : # n_trotters
        self._m = None
        if not W is None :
            self.set_problem(W, b0, b1, optimize)
            self.set_solver_preference(n_trotters)

    def _vars(self) :
        return self._h0, self._h1, self._J, self._c, self._q0, self._q1

    def _Esign(self) :
        return self._optimize.Esign

    def _get_dim(self) :
        return self._dim[0], self._dim[1], self._m

    def _check_dim(self, W, b0, b1) : # FIXME: wrong checks
        if len(b0) != self._dim[0] :
            return False;
        if len(b1) != self._dim[1] :
            return False;
        return True
        
    def set_problem(self, W, b0, b1, optimize = sqaod.minimize) :
        self._dim = (W.shape[1], W.shape[0])
        if not self._check_dim(W, b0, b1) :
            raise Exception('dimension does not match, W: {0}, b0: {1}, b1: {2}.'\
                            .format(W.shape, b0.size, b1.size))

        self._optimize = optimize
        self._h0, self._h1, self._J, self._c = \
            formulas.bipartite_graph_calculate_hJc(b0, b1, W)
        Esign = self._Esign()
        self._h0, self._h1 = self._h0 * Esign, self._h1 * Esign
        self._J, self._c = Esign * self._J, Esign * self._c

    def set_solver_preference(self, n_trotters) :
        # set n_trotters.  The default value assumed to N / 4
        if n_trotters is None :
            n_trotters = (self._dim[0] + self._dim[1]) / 4
        self._m = max(2, n_trotters)
        self._q0 = np.empty((self._m, self._dim[0]), dtype=np.int8)
        self._q1 = np.empty((self._m, self._dim[1]), dtype=np.int8)
        
    def get_E(self) :
        return self._Esign() * np.min(self._E);

    def get_solutions(self) :
        Esign = self._Esign()
        sols = []
        for idx in range(self._m) :
            x0 = common.bits_from_qbits(self._q0[idx])
            x1 = common.bits_from_qbits(self._q1[idx])
            sols.append((Esign * self._E[idx], x0, x1))
        return sols

    def set_x(self, x0, x1) :
        if (x0.shape[len(x0.shape) - 1] != self._dim[0]) \
           or (x1.shape[len(x1.shape) - 1] != self._dim[1]) :
            raise Exception("Dim does not match.")
        q0 = common.bits_to_qbits(x0)
        q1 = common.bits_to_qbits(x1)
        for im in range(0, self._m) :
            self._q0[:][im] = q0[:]
            self._q1[:][im] = q1[:]

    # Ising model / spins
    
    def get_hJc(self) :
        return self._h, self._J, self._c
            
    def get_q(self) :
        return self._q0, self._q1
        
    def set_q(self, q0, q1) :
        if (len(q0) != self._dim[0]) or (len(q0) != self._dim[0]) :
            raise "Dim does not match."
        for im in range(0, self._m) :
            self._q0[:][im] = q0
            self._q1[:][im] = q1

    def randomize_q(self) :
        common.randomize_qbits(self._q0)
        common.randomize_qbits(self._q1)

    def init_anneal(self) :
        if self._m is None :
            self.set_solver_preference(None)
        
    def _anneal_half_step(self, N, qAnneal, h, J, qFixed, G, kT, m) :
        dEmat = np.matmul(J, qFixed.T)
        twoDivM = 2. / m
        tempCoef = np.log(np.tanh(G/kT/m)) / kT
        invKT = 1. / kT
        mrange = range(m);
        random.shuffle(mrange)
        for rim in mrange:
            for iq in range(N):
                q = qAnneal[rim][iq]
                dE = - twoDivM * q * (h[iq] + dEmat[iq, rim])
                mNeibour0 = (rim + m - 1) % m
                mNeibour1 = (rim + 1) % m
                dE -= q * (qAnneal[mNeibour0][iq] + qAnneal[mNeibour1][iq]) * tempCoef
                thresh = 1 if dE < 0 else np.exp(- dE * invKT) 
                if thresh > np.random.rand():
                    qAnneal[rim][iq] = -q
                    
    def anneal_one_step(self, G, kT) :
        h0, h1, J, c, q0, q1 = self._vars()
        N0, N1, m = self._get_dim()
        self._anneal_half_step(N1, q1, h1, J, q0, G, kT, m)
        self._anneal_half_step(N0, q0, h0, J.T, q1, G, kT, m)

    def calculate_E(self) :
        h0, h1, J, c, q0, q1 = self._vars()
        self._E = np.empty((self._m), J.dtype)
        for idx in range(self._m) :
            # FIXME: 1d output for batch calculation
            self._E[idx] = \
                formulas.bipartite_graph_calculate_E_from_qbits(h0, h1, J, c, q0[idx], q1[idx])

def bipartite_graph_annealer(W = None, b0 = None, b1 = None, \
                             optimize = sqaod.minimize, n_trotters = None) :
    return BipartiteGraphAnnealer(W, b0, b1, optimize, n_trotters)


if __name__ == '__main__' :
    N0 = 40
    N1 = 40
    m = 20
    
    np.random.seed(0)
            
    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5

    an = bipartite_graph_annealer(W, b0, b1, sqaod.minimize, m)
    
    Ginit = 5
    Gfin = 0.01
    kT = 0.02
    tau = 0.99
    n_repeat = 10

    for loop in range(0, n_repeat) :
        an.randomize_q()
        G = Ginit
        while Gfin < G :
            an.anneal_one_step(G, kT)
            G = G * tau

        an.calculate_E()
        E = an.get_E()
        sol = an.get_solutions()
        print E, sol
