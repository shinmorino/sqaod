import numpy as np
import random
import sys

class RBMBFSolver :
    
    def __init__(self, N0 = 0, N1 = 0) :
        if not N0 == 0 :
            self.set_problem_size(N0, N1)

    def set_problem_size(self, N0, N1) :
        self.dim = [N0, N1]
        self._xlist = [ np.zeros((N0), dtype=np.int8), np.zeros((N1), dtype=np.int8) ]

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

        self._W = W
        self._blist = [b0, b1]
        
    def get_x(self, idx) :
        return self._xlist[idx]
    
    def get_E(self) :
        return self.E;

    def _get_vars(self) :
        return self._W, self._blist[0], self._blist[1], self._xlist[0], self._xlist[1]

    def search_optimum(self) :
        iMax = 1 << self.dim[0]
        jMax = 1 << self.dim[1]
        print iMax, jMax
        self.E = sys.float_info.max
        W, b0, b1, x0, x1 = self._get_vars()
        x0min = []
        x1min = []

        for i in range(iMax) :
            x0 = [np.int8(i >> pos & 1) for pos in range(N0 - 1,-1,-1)]
            for j in range(jMax) :
                x1 = [np.int8(j >> pos & 1) for pos in range(N1 - 1,-1,-1)]
                Etmp = - np.dot(b0, x0) - np.dot(b1, x1) - np.dot(x1, np.matmul(W, x0))
                if Etmp < self.E :
                    self.E = Etmp
                    print("{0}:{1} Etmp < Emin : {2}".format(i, j, Etmp))
                    x0min, x1min = x0, x1
                    
        self._xlist = [x0min, x1min]

    def calculate_E(self) :
        W, b0, b1, x0, x1 = self._get_vars()
        self.E = - np.dot(b0, x0) - np.dot(b1, x1) - np.dot(x1, np.matmul(W, x0))
        return self.E
        

def rbm_bf_solver(N0 = 0, N1 = 0) :
    return RBMBFSolver(N0, N1)


if __name__ == '__main__' :
    N0 = 10
    N1 = 8
    
    np.random.seed(0)
        
    bf = rbm_bf_solver(N0, N1)
    
    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    bf.set_qubo(W, b0, b1)
    bf.search_optimum()

    x0 = bf.get_x(0) 
    x1 = bf.get_x(1) 
    E = bf.calculate_E()

    print(x0[:], x1[:], E)
