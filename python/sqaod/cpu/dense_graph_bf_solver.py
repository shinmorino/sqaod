import numpy as np
import sqaod
from sqaod.common import checkers
import cpu_dg_bf_solver as dg_bf_solver

class DenseGraphBFSolver :
    
    def __init__(self, W, optimize, dtype) :
        self.dtype = dtype
        self._ext = dg_bf_solver.new_bf_solver(dtype)
        if not W is None :
            self.set_problem(W, optimize)
            
    def __del__(self) :
        dg_bf_solver.delete_bf_solver(self._ext, self.dtype)

    def set_problem(self, W, optimize = sqaod.minimize) :
        checkers.dense_graph.qubo(W)
        W = sqaod.clone_as_ndarray(W, self.dtype)
        self._N = W.shape[0]
        dg_bf_solver.set_problem(self._ext, W, optimize, self.dtype)
        self._optimize = optimize

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return dg_bf_solver.get_E(self._ext, self.dtype)

    def get_x(self) :
        return dg_bf_solver.get_x(self._ext, self.dtype)

    def init_search(self) :
        dg_bf_solver.init_search(self._ext, self.dtype);
        
    def fin_search(self) :
        dg_bf_solver.fin_search(self._ext, self.dtype);
        
    def search_range(self, iBegin0, iEnd0, iBegin1, iEnd1) :
        dg_bf_solver.search_range(self._ext, iBegin0, iEnd0, iBegin1, iEnd1, self.dtype)
        
    def search(self) :
        N = self._N
        iMax = 1 << N
        iStep = min(256, iMax)
        self.init_search()
        for iTile in range(0, iMax, iStep) :
            dg_bf_solver.search_range(self._ext, iTile, iTile + iStep, self.dtype)
        self.fin_search()
        
    def _search(self) :
        # one liner.  does not accept ctrl+c.
        dg_bf_solver.search(self._ext, self.dtype)


def dense_graph_bf_solver(W = None, optimize = sqaod.minimize, dtype=np.float64) :
    return DenseGraphBFSolver(W, optimize, dtype)


if __name__ == '__main__' :

    np.random.seed(0)
    dtype = np.float32
    N = 8
    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])

#    N = 20
#    W = sqaod.generate_random_symmetric_W(N, -0.5, 0.5, dtype);
    bf = dense_graph_bf_solver(W, sqaod.minimize, dtype)
    bf.search()
    x = bf.get_x() 
    E = bf.get_E()
    print E
    print x

#    import sqaod.py.formulas
#    E = np.empty((1), dtype)
#    for bits in x :
#        print sqaod.py.formulas.dense_graph_calculate_E(W, bits)
