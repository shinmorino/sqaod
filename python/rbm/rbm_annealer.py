import numpy as np
import random


class RBMAnnealer :
    
    def __init__(self, N0 = 0, N1 = 0, m = 0) :
        if not N0 == 0 :
            self.set_problem_size(N0, N1, m)

    def _get_vars(self) :
        return self._J, self._hlist[0], self._hlist[1], self._qlist[0], self._qlist[1]

    def _get_dim(self) :
        return self.dim[0], self.dim[1], self.m
            
    def set_problem_size(self, N0, N1, m) :
        self.dim = [N0, N1]
        self._qlist = [ np.zeros((m, N0), dtype=np.int8), np.zeros((m, N1), dtype=np.int8) ]
        self.m = m;

    def _check_dim(self, W, b0, b1) :
        dimRev = [n for n in reversed(W.shape)]
        if not np.allclose(dimRev, self.dim) :
            return False
        if len(b0) != self.dim[0] :
            return False;
        if len(b1) != self.dim[1] :
            return False;
        return True
        
    def set_qubo(self, W, b0, b1) :
        if self.dim[0] == 0 or self.dim[1] == 0 :
            shape = W.shape()
            self.set_problem_size(shape[0], shape[1], m)
        else :
            if not self._check_dim(W, b0, b1) :
                raise Exception('dimension does not match, W: {0}, b0: {1}, b1: {2}.'\
                                .format(W.shape, b0.size, b1.size))

        self._c = 0.25 * np.sum(W) + 0.5 * (np.sum(b0) + np.sum(b1))
        self._J = 0.25 * W
        N0, N1, m = self._get_dim()
        h0 = [(1. / 4.) * np.sum(W[:, i]) + 0.5 * b0[i] for i in range(0, N0)]
        h1 = [(1. / 4.) * np.sum(W[j]) + 0.5 * b1[j] for j in range(0, N1)]
        self._hlist = [h0, h1]

    def get_x(self, idx) :
        q = self._qlist[idx][0]
        x = np.ndarray([self.dim[idx]], np.int8)
        for i in range(self.dim[idx]) :
            x[i] = 1 if q[i] == 1 else 0
        return x

    def set_x(self, idx, x) :
        if len(x) != self.dim[idx] :
            raise "Error"
        q_int8 = [-1 if xi == 0 else 1 for xi in x]
        for im in range(0, self.m) :
            self._qlist[idx][:][im] = q_int8[:]

        
    def set_ising_model(self, J, h0, h1, c = 0.) :
        if self.dim[0] == 0 or self.dim[1] == 0 :
            shape = J.shape()
            self.set_problem_size(shape[0], shape[1], m)
        else :
            if not self._check_dim(J, h0, h1) :
                raise Exception('dimension does not match, J: {0}, h0: {1}, h1: {2}.'\
                                .format(W.shape, b0.size, b1.size))
        self._J = J;
        self._hlist = [h0, h1]
        self._c = c
        
    def get_q(self, idx) :
        return self._qlist[idx]
        
    def set_q(self, idx, q) :
        if len(q) != self.dim[idx] :
            raise "Error"
        q_int8 = q.astype(np.int8)
        for im in range(0, self.m) :
            self._qlist[idx][:][im] = q_int8[:]

    def randomize_q(self, idx) :
        q = self._qlist[idx];
        for v in np.nditer(q, [], ['readwrite']):
            v[...] = np.random.choice([-1,1])
    
    def get_hJc(self) :
        return self._h, self._J, self._c
        
    def get_E(self) :
        return self.E;

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
                dE = twoDivM * q * (h[iq] + dEmat[iq, rim])
                mNeibour0 = (rim + m - 1) % m
                mNeibour1 = (rim + 1) % m
                dE += -q * (qAnneal[mNeibour0][iq] + qAnneal[mNeibour1][iq]) * tempCoef
                thresh = 1 if dE < 0 else np.exp(- dE * invKT) 
                if thresh > np.random.rand():
                    qAnneal[rim][iq] = -q
                    
    def anneal_one_step(self, G, kT) :
        J, h0, h1, q0, q1 = self._get_vars()
        N0, N1, m = self._get_dim()
        self._anneal_half_step(N1, q1, h1, J, q0, G, kT, m)
        self._anneal_half_step(N0, q0, h0, J.T, q1, G, kT, m)

    def calculate_E(self) :
        J, h0, h1, q0, q1 = self._get_vars()
        self.E = - np.dot(h0, q0[0]) - np.dot(h1, q1[0]) - np.dot(q1[0], np.matmul(J, q0[0])) - self._c
        return self.E
        

def rbm_annealer(N0 = 0, N1 = 0, m = 0) :
    return RBMAnnealer(N0, N1, m)


if __name__ == '__main__' :
    N0 = 40
    N1 = 40
    m = 20

    
    np.random.seed(0)
        
    an = rbm_annealer(N0, N1, m)
    
    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    an.set_qubo(W, b0, b1)
    #an.set_ising_model(W, b0, b1)
    
    Ginit = 5
    Gfin = 0.01
    kT = 0.02
    tau = 0.99
    n_repeat = 10

    for loop in range(0, n_repeat) :
        an.randomize_q(0)
        an.randomize_q(1)
        G = Ginit
        while Gfin < G :
            an.anneal_one_step(G, kT)
            G = G * tau

        q0 = an.get_q(0) 
        q1 = an.get_q(1) 
        E = an.calculate_E()
        print(q0[0,:], q1[0,:], E)
