import numpy as np
import random
import solver_traits
import utils
import sys
import py
import tags

class DenseGraphBFSolver :
    
    def __init__(self, N = 0) :
        self._verbose = False
        self.set_problem_size(N)
        
    def set_problem_size(self, N) :
        self._N = N
        self._x = np.zeros((N), dtype=np.int8)
        
    def set_problem(self, W, optimize = tags.minimize) :
        if self._N == 0 :
            self.set_problem_size(W.shape[0])
        self._W = W.copy()
        if optimize is tags.minimize :
            self._W = self._W * -1.
            self.minimize = True
        else :
            self.minimize = False
            
    def _get_vars(self) :
        return self._W, self._x

    def get_W(self) :
        return self._W

    def get_x(self) :
        return self._x

    def get_E(self) :
        return self.E;

    def calculate_E(self) :
        h, J, c, q = self._get_vars()
        self.E = solver_traits.dense_graph_calculate_E_from_qbits(h, J, c, q[0])

    def _search_optimum_batched(self) :
        N = self._N
        iMax = 1 << N
        self.E = sys.float_info.max
        W, x = self._get_vars()
        xmin = []

        iStep = min(256, iMax)
        for iTile in range(0, iMax, iStep) :
            x = utils.create_bits_sequence(range(iTile, iTile + iStep), N)
            Etmp = solver_traits.dense_graph_batch_calculate_E(W, x)
            for i in range(iStep) :
                if Etmp[i] < self.E :
                    self.E = Etmp[i]
                    xmin = np.copy(x[i])
                    if self._verbose :
                        print("{0} Etmp < Emin : {1}".format(iTile + i, Etmp))

        self._x = xmin
        if not self.minimize :
            self.E = - self.E
        

def dense_graph_bf_solver(N = 0) :
    return DenseGraphBFSolver(N)


if __name__ == '__main__' :

    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    
    
    N = 8
    bf = dense_graph_bf_solver(8)

    bf.set_problem(W, tags.minimize)
    bf._search_optimum_batched()
    x = bf.get_x() 
    E = bf.get_E()
    print(E, x)

    bf.set_problem(W, tags.maximize)
    bf._search_optimum_batched()
    x = bf.get_x() 
    E = bf.get_E()
    print(E, x)
