import numpy as np
import random

class RBMAnnealer :
    
    def __init__(self, N0 = 0, N1 = 0, m = 0) :
        if not N0 == 0 :
            self.set_problem_size(N0, N1, m)

    def set_problem_size(self, N0, N1, m) :
        self.dim = [N0, N1]
        self._qlist = [ np.zeros((m, N0), dtype=np.int8), np.zeros((m, N1), dtype=np.int8) ]
        self.m = m;

    def _check_dim(self, W, b0, b1) :
        if W.shape != (N1, N0) :
            return False
        if len(b0) != N0 :
            return False;
        if len(b1) != N1 :
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

        self._c = (1. / 4.) * np.sum(W) + (1. / 2.) * (np.sum(b0) + np.sum(b1))
        self._J = W * (1. /  4.)
        h0 = [np.sum(W[:, i]) - 0.5 * b0[i] for i in range(0, N0)]
        h1 = [np.sum(W[i]) - 0.5 * b1[i] for i in range(0, N1)]
        self._hlist = [h0, h1]
        
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
        for i in range(0, m) :
            self._qlist[i][:] = q_int8

    def randomize_q(self, idx) :
        q = self._qlist[idx];
        for v in np.nditer(q, [], ['readwrite']):
            v[...] = np.random.choice([-1,1])
    
    def get_J(self) :
        return self._J

    def get_h(self, idx) :
        return self._hlist[idx]
        
    def get_E(self) :
        return self.E;

    def _anneal_half_step(self, N, qAnneal, h, J, qFixed, G, kT) :
        dEmat = np.matmul(J, qFixed.T)
        twoDivM = 2 / m
        tempCoef = np.log(np.tanh(G/kT/m)) / kT
        invKT = 1. / kT
        for im in range(self.m):
            for iq in range(N):
                q = qAnneal[im][iq]
                dE = twoDivM * q * (h[iq] + dEmat[iq, im])
                mNeibour0 = (im + m - 1) % m
                mNeibour1 = (im + 1) % m
                dE += -q * (qAnneal[mNeibour0][iq] + qAnneal[mNeibour1][iq]) * tempCoef
                thresh = 1 if dE < 0 else np.exp(- dE * invKT) 
                if thresh > np.random.rand():
                    qAnneal[im][iq] = -q

    def _get_vers(self) :
        return self._J, self._hlist[0], self._hlist[1], self._qlist[0], self._qlist[1]
                    
    def anneal_one_step(self, G, kT) :
        J, h0, h1, q0, q1 = self._get_vers()
        self._anneal_half_step(N1, q1, h1, J, q0, G, kT)
        self._anneal_half_step(N0, q0, h0, np.transpose(J), q1, G, kT)

    def calculate_E(self) :
        J, h0, h1, q0, q1 = self._get_vers()
        J = self._J
        self.E = np.dot(h0, q0[0]) + np.dot(h1, q1[0]) + np.dot(h1, np.matmul(J, q0[0]))
        

def rbm_annealer(N0 = 0, N1 = 0, m = 0) :
    return RBMAnnealer(N0, N1, m)


if __name__ == '__main__' :
    N0 = 10
    N1 = 8
    m = 5
    
    np.random.seed(10000)
        
    an = rbm_annealer(N0, N1, m)
    W = 2 * np.random.random((N1, N0)) - 1.
    b0 = 2 * np.random.random((N0)) - 1.
    b1 = 2 * np.random.random((N1)) - 1.
    
    an.set_qubo(W, b0, b1)
    an.randomize_q(0)
    an.randomize_q(1)

    Ginit = 5.
    Gfin = 0.01

    nRepeat = 4
    kT = 0.02
    tau = 0.99

    for loop in range(0, nRepeat) :
        G = Ginit
        while Gfin < G :
            an.anneal_one_step(G, kT)
            an.calculate_E()
            G = G * tau

    q0 = an.get_q(0) 
    q1 = an.get_q(1) 
    E = an.get_E() 
    print(q0[0,:], q1[0,:], E)
    
