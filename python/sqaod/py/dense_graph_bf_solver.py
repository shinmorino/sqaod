import numpy as np
import random
import sys
import sqaod
import sqaod.utils as utils
import solver_traits

class DenseGraphBFSolver :
    
    def __init__(self, W = None, optimize = sqaod.minimize) :
        self._verbose = False
        if W is not None :
            self.set_problem(W, optimize)

    def _Esign(self) :
        return self._optimize.Esign
            
    def set_problem(self, W, optimize = sqaod.minimize) :
        # FIXME: check W dims, is symmetric ? */
        self._W = W.copy()
        N = W.shape[0]
        self._N = N
        self._x = np.zeros((N), dtype=np.int8)
        self._optimize = optimize
        self._W = W * optimize.Esign
            
    def get_E(self) :
        return self._Esign() * self._Emin
    
    def get_solutions(self) :
        return self._solutions

    def _reset_solutions(self, Etmp, x) :
        self._solutions = [(self._Esign() * Etmp, x)]
        if self._verbose :
            print("Enew < E : {1}".format(Etmp))

    def _append_to_solutions(self, Etmp, x) :
        self._solutions.append((self._Esign() * Etmp, x))
        if self._verbose :
            print("{0} Etmp < Emin : {1}".format(iTile + i, Etmp))

    def _search_optimum_batched(self) :
        N = self._N
        iMax = 1 << N
        self._Emin = sys.float_info.max
        W = self._W
        self._solutions = []

        iStep = min(256, iMax)
        for iTile in range(0, iMax, iStep) :
            x = utils.create_bits_sequence(range(iTile, iTile + iStep), N)
            Etmp = solver_traits.dense_graph_batch_calculate_E(W, x)
            for i in range(iStep) :
                if self._Emin < Etmp[i] :
                    continue
                elif Etmp[i] < self._Emin :
                    self._Emin = Etmp[i]
                    self._reset_solutions(Etmp[i], np.copy(x[i]))
                else :
                    self._append_to_solutions(self._Emin, x[i])

def dense_graph_bf_solver(W = None, optimize = sqaod.minimize) :
    return DenseGraphBFSolver(W, optimize)


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
    bf = dense_graph_bf_solver(W, sqaod.minimize)
    bf._search_optimum_batched()
    E = bf.get_E()
    x = bf.get_solutions() 
    print E, len(x), x[0]

    bf = dense_graph_bf_solver(W, sqaod.maximize)
    bf._search_optimum_batched()
    E = bf.get_E()
    x = bf.get_solutions() 
    print E, len(x), x[0]
