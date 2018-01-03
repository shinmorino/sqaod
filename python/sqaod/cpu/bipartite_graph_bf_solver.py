import numpy as np
import random
import sys
import sqaod
from sqaod.common import checkers
import cpu_bg_bf_solver as bg_bf_solver


class BipartiteGraphBFSolver :
    
    def __init__(self, b0, b1, W, optimize, dtype) :
        self.dtype = dtype
        self._ext = bg_bf_solver.new_solver(dtype)
        if not W is None :
            self.set_problem(b0, b1, W, optimize)
            
    def __del__(self) :
        bg_bf_solver.delete_solver(self._ext, self.dtype)

    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)
        b0, b1, W = sqaod.clone_as_ndarray_from_vars([b0, b1, W], self.dtype)
        self._dim = (b0.shape[0], b1.shape[0])
        bg_bf_solver.set_problem(self._ext, b0, b1, W, optimize, self.dtype)
        self._optimize = optimize

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return bg_bf_solver.get_E(self._ext, self.dtype);

    def get_x(self) :
        return bg_bf_solver.get_x(self._ext, self.dtype);
    
    def init_search(self) :
        bg_bf_solver.init_search(self._ext, self.dtype);
        
    def fin_search(self) :
        bg_bf_solver.fin_search(self._ext, self.dtype);
        
    def search_range(self, iBegin0, iEnd0, iBegin1, iEnd1) :
        bg_bf_solver.search_range(self._ext, iBegin0, iEnd0, iBegin1, iEnd1, self.dtype)
        
    def _search(self) :
        # one liner.  does not accept ctrl+c.
        bg_bf_solver.search(self._ext, self.dtype);
        

    def search(self) :
        nStep = 1024
        self.init_search()

        N0, N1 = self._dim[0], self._dim[1]
        iMax = 1 << N0
        jMax = 1 << N1

        iStep = min(nStep, iMax)
        jStep = min(nStep, jMax)
        for iTile in range(0, iMax, iStep) :
            for jTile in range(0, jMax, jStep) :
                self.search_range(iTile, iTile + iStep, jTile, jTile + jStep)
        
        self.fin_search()

        

def bipartite_graph_bf_solver(b0 = None, b1 = None, W = None, optimize = sqaod.minimize, dtype = np.float64) :
    return BipartiteGraphBFSolver(b0, b1, W, optimize, dtype)


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
    print E.shape, E
    print x
