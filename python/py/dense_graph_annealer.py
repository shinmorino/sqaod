import numpy as np
import random
import dense_graph_traits
import utils

class DenseGraphAnnealer :
    
    def __init__(self, N = 0, m = 0) :
        if not N == 0 :
            self.set_problem_size(N, m)

    def set_seed(seed) :
        random.seed(seed)
        
    def set_problem_size(self, N, m) :
        self.N = N
        self.m = m;
        self.q = np.zeros((m, N), dtype=np.int8)

    def _get_vars(self) :
        return self.h, self.J, self.c, self.q
        
    def set_qubo(self, W) :
        self.h, self.J, self.c = dense_graph_traits.calculate_hJc(W)

    def randomize_q(self) :
        utils.randomize_qbits(self.q)

    def get_q(self) :
        return self.q

    def get_hJc(self) :
        return h, J, c

    def get_E(self) :
        return self.E;

    def calculate_E(self) :
        h, J, c, q = self._get_vars()
        self.E = dense_graph_traits.calculate_E_from_hJc(h, J, c, q[0])
            
    def anneal_one_step(self, G, kT) :
        h, J, c, q = self._get_vars()
        N = self.N
        m = self.m
        two_div_m = 2. / np.float64(m)
        coef = np.log(np.tanh(G/kT/m)) / kT
        
        for i in range(self.N * self.m):
            x = np.random.randint(N)
            y = np.random.randint(m)
            sum = np.dot(J[x], q[y]); # diagnoal elements in J are zero.
            qyx = q[y][x]
            dE = two_div_m * qyx * (h[x] + sum)
            dE += - qyx * (q[(m + y - 1) % m][x] + q[(y + 1) % m][x]) * coef
            if np.exp(-dE / kT) > np.random.rand():
                q[y][x] = - qyx
        

def dense_graph_annealer(N = 0, m = 0) :
    return DenseGraphAnnealer(N, m)


if __name__ == '__main__' :

    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    
    
    ann = dense_graph_annealer(8, 4)
    
    ann.set_qubo(W)
    
    Ginit = 5.
    Gfin = 0.01
    
    N = 8
    m = 4
    
    nRepeat = 4
    kT = 0.02
    tau = 0.99
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.randomize_q()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        ann.calculate_E()
        q = ann.get_q() 
        E = ann.get_E()
        print(q, E)
