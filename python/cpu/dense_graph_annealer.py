import numpy as np
import native

class SimpleNativeAnnealer :

    def __init__() :
        pass
    
    def __init__(self, N, m) :
        self.set_problem_size(N, m)

    def set_seed(seed) :
        native.set_seed(seed)
        
    def set_problem_size(self, N, m) :
        self.N = N
        self.m = m;
        self.q = np.zeros((m, N), dtype=np.int8)
    
    def set_qubo(self, qubo) :
        if qubo.dtype != 'float64' :
            qubo = qubo.astype(np.float64)
        self.h, self.J, self.c = native.create_hJc(qubo)

    def get_q(self) :
        return self.q

    def get_hJc(self) :
        return h, J, c

    def get_E(self) :
        return self.E;
    
    def randomize_q(self) :
        native.randomize_q(self.q)
    
    def anneal_one_step(self, G, kT) :
        self.E = native.anneal_one_step(self.q, G, kT, self.h, self.J, self.c)


def simple_native_annealer(N = 0, m = 0) :
    return SimpleNativeAnnealer(N, m)
