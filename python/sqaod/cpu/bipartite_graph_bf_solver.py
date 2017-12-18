import numpy as np
import random
import sys
import sqaod
import sqaod.common as common
import cpu_bg_bf_solver as bf_solver

class BipartiteGraphBFSolver :
    
    def __init__(self, W, b0, b1, optimize, dtype) :
        self.dtype = dtype
        self._ext = bf_solver.new_bf_solver(dtype)
        if not W is None :
            self.set_problem(b0, b1, W, optimize)
            
    def __del__(self) :
        bf_solver.delete_bf_solver(self._ext, self.dtype)

    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        # FIXME: add error check.
        self._dim = (b0.shape[0], b1.shape[0])
        bf_solver.set_problem(self._ext, b0, b1, W, optimize, self.dtype)

    def get_E(self) :
        return bf_solver.get_E(self._ext, self.dtype);
        # return self._Esign() * self._Emin

    def get_solutions(self) :
        return bf_solver.get_x(self._ext, self.dtype);
        #return self._solutions
    
    def _search(self) :
        bf_solver.search(self._ext, self.dtype);
        
    def _search_range(self) :
        N0, N1 = self._dim[0], self._dim[1]
        iMax = 1 << N0
        jMax = 1 << N1

        iStep = min(256, iMax)
        jStep = min(256, jMax)
        for iTile in range(0, iMax, iStep) :
            for jTile in range(0, jMax, jStep) :
                bf_solver.search_range(iTile, iTile + iStep,
                                       jTile, jTile + jStep)

    def search(self) :
        self._search()
        

def bipartite_graph_bf_solver(b0 = None, b1 = None, W = None, optimize = sqaod.minimize, dtype=np.float64) :
    return BipartiteGraphBFSolver(b0, b1, W, optimize, dtype)


if __name__ == '__main__' :
    N0 = 14
    N1 = 5
    
    np.random.seed(0)

    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    
    bf = bipartite_graph_bf_solver(W, b0, b1)
    bf.search()
    E = bf.get_E()
    solutions = bf.get_solutions() 
    print E
    print solutions

