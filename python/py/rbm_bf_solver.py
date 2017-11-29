import numpy as np
import random
import sys

class RBMBFSolver :
    
    def __init__(self, N0 = 0, N1 = 0, verbose = False) :
        if not N0 == 0 :
            self.set_problem_size(N0, N1)
        self._verbose = False

    def _get_vars(self) :
        return self._W, self._blist[0], self._blist[1], self._xlist[0], self._xlist[1]

    def _get_dim(self) :
        return self.dim[0], self.dim[1]

    def set_problem_size(self, N0, N1) :
        self.dim = [N0, N1]
        self._xlist = [ np.zeros((N0), dtype=np.int8), np.zeros((N1), dtype=np.int8) ]
        
    def _check_dim(self, W, b0, b1) :
        N0, N1 = self._get_dim()
        dimRev = [n for n in reversed(W.shape)]
        if not np.allclose(dimRev, self.dim) :
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

    def _search_optimum(self) :
        N0, N1 = self._get_dim()
        iMax = 1 << N0
        jMax = 1 << N1
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
                    x0min, x1min = np.copy(x0), np.copy(x1)
                    if self._verbose :
                        print("{0}:{1} Etmp < Emin : {2}".format(i, j, Etmp))
                    
        self._xlist = [x0min, x1min]
        
    @staticmethod
    def _create_bit_sequences(v0, v1, N) :
        x = np.ndarray((v1 - v0, N), np.int8)
        for v in range(v0, v1) :
            for pos in range(N - 1, -1, -1) :
                x[v - v0][pos] = np.int8(v >> pos & 1)
        return x

        
    def _search_optimum_batched(self) :
        N0, N1 = self._get_dim()
        iMax = 1 << N0
        jMax = 1 << N1
        self.E = sys.float_info.max
        W, b0, b1, _, _ = self._get_vars()
        x0min = []
        x1min = []

        iStep = min(256, iMax)
        jStep = min(256, jMax)
        for iTile in range(0, iMax, iStep) :
            x0 = RBMBFSolver._create_bit_sequences(iTile, iTile + iStep, N0)
            for jTile in range(0, jMax, jStep) :
                x1 = RBMBFSolver._create_bit_sequences(jTile, jTile + jStep, N1)
                Etmp = - np.matmul(b0, x0.T).reshape(1, iStep) \
                       - np.matmul(b1, x1.T).reshape(jStep, 1) - np.matmul(x1, np.matmul(W, x0.T))

                for j in range(jStep) :
                    for i in range(iStep) :
                        if Etmp[j][i] < self.E :
                            self.E = Etmp[j][i]
                            x0min, x1min = np.copy(x0[i]), np.copy(x1[j])
                            if self._verbose :
                                print("{0}:{1} Etmp < Emin : {2}".format(i, j, Etmp))
                    
        self._xlist = [x0min, x1min]


    def search_optimum(self) :
        self._search_optimum_batched()
                              
    def calculate_E(self) :
        W, b0, b1, x0, x1 = self._get_vars()
        self.E = - np.dot(b0, x0) - np.dot(b1, x1) - np.dot(x1, np.matmul(W, x0))
        return self.E
        

def rbm_bf_solver(N0 = 0, N1 = 0) :
    return RBMBFSolver(N0, N1)


if __name__ == '__main__' :
    N0 = 14
    N1 = 5
    
    np.random.seed(0)
        
    bf = rbm_bf_solver(N0, N1)
    
    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    bf.set_qubo(W, b0, b1)
    bf._search_optimum()
    x0 = bf.get_x(0) 
    x1 = bf.get_x(1) 
    E = bf.calculate_E()
    print(x0[:], x1[:], E)
    
    bf._search_optimum_batched()
    x0 = bf.get_x(0) 
    x1 = bf.get_x(1) 
    E = bf.calculate_E()
    print(x0[:], x1[:], E)

