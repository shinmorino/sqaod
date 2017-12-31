import numpy as np
import sys
import sqaod
from sqaod.common import checkers

class BipartiteGraphBFSolver :
    
    def __init__(self, W, b0, b1, optimize) :
        self._verbose = False
        if not W is None :
            self.set_problem(W, b0, b1, optimize)

    def _vars(self) :
        return self._W, self._blist[0], self._blist[1]

    def _get_dim(self) :
        return self._dim[0], self._dim[1]
    
    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)
        self._dim = (b0.shape[0], b1.shape[0])
        N0, N1 = self._get_dim()
        self._pairs = []
        self._optimize = optimize
        self._W = self._optimize.sign(W)
        self._blist = (self._optimize.sign(b0), self._optimize.sign(b1))

    def get_optimize_dir(self) :
        return self._optimize
    
    def get_E(self) :
        return self._E

    def get_x(self) :
        return self._x_pairs

    def init_search(self) :
        N0, N1 = self._get_dim()
        self._iMax = 1 << N0
        self._jMax = 1 << N1
        self._Emin = sys.float_info.max

    def fin_search(self) :
        nX = len(self._x_pairs)
        self._E = np.empty((nX))
        self._E[...] = self._optimize.sign(self._Emin)

    # Not used.  Keeping it for reference.    
    def _search_simple(self) :
        N0, N1 = self._get_dim()
        W, b0, b1 = self._vars()

        for i in range(self._iMax) :
            x0 = sqaod.create_bits_sequence((i), N0)
            for j in range(self._jMax) :
                x1 = sqaod.create_bits_sequence((j), N1)
                Etmp = np.dot(b0, x0.transpose()) + np.dot(b1, x1.transpose()) \
                       + np.dot(x1, np.matmul(W, x0.transpose()))
                if self._Emin < Etmp :
                    continue
                elif Etmp < self._Emin :
                    self._Emin = Etmp
                    self._x_pairs = [(x0, x1)]
                else :
                    self._x_pairs.append = [(x0, x1)]
        
    def search_range(self, iBegin, iEnd, jBegin, jEnd) :
        N0, N1 = self._get_dim()
        W, b0, b1 = self._vars()
        iBegin = max(0, min(self._iMax, iBegin))
        iEnd = max(0, min(self._iMax, iEnd))
        jBegin = max(0, min(self._jMax, jBegin))
        jEnd = max(0, min(self._jMax, jEnd))

        iStep = iEnd - iBegin
        jStep = jEnd - jBegin

        x0 = sqaod.create_bits_sequence(range(iBegin, iEnd), N0)
        x1 = sqaod.create_bits_sequence(range(jBegin, jEnd), N1)
        Etmp = np.matmul(b0, x0.T).reshape(1, iStep) \
               + np.matmul(b1, x1.T).reshape(jStep, 1) + np.matmul(x1, np.matmul(W, x0.T))
        for j in range(jEnd - jBegin) :
            for i in range(iEnd - iBegin) :
                if self._Emin < Etmp[j][i] :
                    continue
                elif Etmp[j][i] < self._Emin :
                    self._Emin = Etmp[j][i]
                    self._x_pairs = [(x0[i], x1[j])]
                else :
                    self._x_pairs.append((x0[i], x1[j]))
                    
    def search(self) :
        self.init_search()
        
        iStep = min(256, self._iMax)
        jStep = min(256, self._jMax)
        for j in range(0, self._jMax, jStep) :
            for i in range(0, self._iMax, iStep) :
                self.search_range(i, i + iStep, j, j + jStep)

        self.fin_search()
        
def bipartite_graph_bf_solver(b0 = None, b1 = None, W = None, optimize = sqaod.minimize) :
    return BipartiteGraphBFSolver(b0, b1, W, optimize)


if __name__ == '__main__' :
    N0 = 14
    N1 = 5
    
    np.random.seed(0)

    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    
    bf = bipartite_graph_bf_solver(b0, b1, W)
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print E
    print x
