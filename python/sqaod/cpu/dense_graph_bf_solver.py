import numpy as np
import sqaod
import sqaod.utils as utils
import solver_traits
import cpu_dg_bf_solver as dg_bf_solver

class DenseGraphBFSolver :
    
    def __init__(self, N, dtype) :
        self.dtype = dtype
        self.ext = dg_bf_solver.new_bf_solver(dtype)
        self.set_problem_size(N)
        
    def set_problem_size(self, N) :
        self._N = N;
        dg_bf_solver.set_problem_size(self.ext, N, self.dtype)
        
    def set_problem(self, W, optimize = sqaod.minimize) :
        if self._N == 0 :
            self.set_problem_size(self.ext, W.shape[0])
        W = solver_traits.clone_as_np_buffer(W, self.dtype)
        dg_bf_solver.set_problem(self.ext, W, optimize, self.dtype)

    def get_x(self) :
        return dg_bf_solver.get_x(self.ext, self.dtype)

    def get_E(self) :
        return dg_bf_solver.get_E(self.ext, self.dtype)

    def search(self) :
        N = self._N
        iMax = 1 << N
        iStep = min(256, iMax)
        for iTile in range(0, iMax, iStep) :
            dg_bf_solver.search_range(self.ext, iTile, iTile + iStep, self.dtype)

    def _search(self) :
        dg_bf_solver.search(self.ext, self.dtype)


def dense_graph_bf_solver(N = 0, dtype=np.float64) :
    return DenseGraphBFSolver(N, dtype)


if __name__ == '__main__' :

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
    bf = dense_graph_bf_solver(N, np.float32)
    np.random.seed(0)
    W = utils.generate_random_symmetric_W(N, -0.5, 0.5, np.float32);

    bf.set_problem(W, sqaod.minimize)
    bf.search()
    x = bf.get_x() 
    E = bf.get_E()
    print(E, x)
