import numpy as np
import sqaod
import sqaod.common as common
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
        # FIXME: check W shape.
        W = common.clone_as_np_buffer(W, self.dtype)
        self._N = W.shape[0]
        dg_bf_solver.set_problem(self._ext, W, optimize, self.dtype)

    def get_x(self) :
        E = self.get_E()
        solutions = []
        x = dg_bf_solver.get_x(self._ext, self.dtype)
        for i in range(x.shape[0]) :
            solutions.append((E, x[i,:]))
        return solutions

    def get_E(self) :
        return dg_bf_solver.get_E(self._ext, self.dtype)

    def search(self) :
        N = self._N
        iMax = 1 << N
        iStep = min(256, iMax)
        dg_bf_solver.init_search(self._ext, self.dtype)
        for iTile in range(0, iMax, iStep) :
            dg_bf_solver.search_range(self._ext, iTile, iTile + iStep, self.dtype)

    def _search(self) :
        dg_bf_solver.search(self._ext, self.dtype)


def dense_graph_bf_solver(W = None, optimize = sqaod.minimize, dtype=np.float64) :
    return DenseGraphBFSolver(W, optimize, dtype)


if __name__ == '__main__' :

    np.random.seed(0)

    N = 8
    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])

    N = 20
    W = common.generate_random_symmetric_W(N, -0.5, 0.5, np.float32);
    bf = dense_graph_bf_solver(W, sqaod.minimize, np.float32)
    bf.search()
    x = bf.get_x() 
    E = bf.get_E()
    print E
    print x
