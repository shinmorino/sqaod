import numpy as np
import random
import sys
import sqaod
import sqaod.common as common
import formulas

class DenseGraphBFSolver :
    
    def __init__(self, W = None, optimize = sqaod.minimize) :
        if W is not None :
            self.set_problem(W, optimize)

    def _Esign(self) :
        return self._optimize.Esign
            
    def set_problem(self, W, optimize = sqaod.minimize) :
        # FIXME: check W dims, is symmetric ? */
        self._W = W.copy()
        N = W.shape[0]
        self._N = N
        self._x = []
        self._optimize = optimize
        self._W = W * optimize.Esign
            
    def get_E(self) :
        return self._E
    
    def get_x(self) :
        return self._x

    def init_search(self) :
        N = self._N
        self._xMax = 1 << N
        self._Emin = sys.float_info.max

    def fin_search(self) :
        nMinX = len(self._x)
        self._E = np.empty((nMinX))
        self._E[...] = self._Esign() * self._Emin

    def search_range(self, xBegin, xEnd) :
        N = self._N
        W = self._W
        xBegin = max(0, min(self._xMax, xBegin))
        xEnd = max(0, min(self._xMax, xEnd))
        x = common.create_bits_sequence(range(xBegin, xEnd), N)
        Etmp = formulas.dense_graph_batch_calculate_E(W, x)
        for i in range(xEnd - xBegin) :
            if self._Emin < Etmp[i] :
                continue
            elif Etmp[i] < self._Emin :
                self._Emin = Etmp[i]
                self._x = [x[i]]
            else :
                self._x.append(x[i])

    def search(self) :
        self.init_search()
        iStep = min(256, self._xMax)
        for iTile in range(0, self._xMax, iStep) :
            self.search_range(iTile, iTile + iStep)
        self.fin_search()
        
    
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
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print E, len(x), x[0]

    bf = dense_graph_bf_solver(W, sqaod.maximize)
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print E, len(x), x[0]
