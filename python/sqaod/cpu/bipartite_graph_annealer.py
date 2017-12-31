import numpy as np
import sqaod
from sqaod.common import checkers
import formulas
import cpu_bg_annealer as bg_annealer


class BipartiteGraphAnnealer :

    def __init__(self, b0, b1, W, optimize, n_trotters, dtype) : # n_trotters
        self.dtype = dtype
        self._ext = bg_annealer.new_annealer(dtype)
        if not W is None :
            self.set_problem(b0, b1, W, optimize)
        if not n_trotters is None :
            self.set_solver_preference(n_trotters)

    def __del__(self) :
        bg_annealer.delete_annealer(self._ext, self.dtype)
        
    def rand_seed(self, seed) :
        bg_annealer.rand_seed(self._ext, seed, self.dtype)
            
    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)
        bg_annealer.set_problem(self._ext, b0, b1, W, optimize, self.dtype);
        self._optimize = optimize

    def get_optimize_dir(self) :
        return self._optimize

    def get_problem_size(self) :
        return bg_annealer.get_problem_size(self._ext, self.dtype)

    def set_solver_preference(self, n_trotters) :
        # set n_trotters.  The default value assumed to N / 4
        N0, N1, m = self.get_problem_size()
        bg_annealer.set_solver_preference(self._ext, n_trotters, self.dtype);
        N0, N1, m = self.get_problem_size()
        self._E = np.empty((m), self.dtype)
        
    def get_E(self) :
        return self._E

    def get_x(self) :
        return bg_annealer.get_x(self._ext, self.dtype)

    def set_x(self, x0, x1) :
        bg_annealer.set_x(self._ext, x0, x1, self.dtype)

    # Ising model / spins
    
    def get_hJc(self) :
        N0, N1, m = self._get_dim()
        h0 = np.ndarray((N0), self.dtype);
        h1 = np.ndarray((N1), self.dtype);
        J = np.ndarray((N1, N0), self.dtype);
        c = np.ndarray((1), self.dtype)
        bg_annealer.get_hJc(self._ext, h0, h1, J, c, self.dtype)
        return h0, h1, J, c[0]
            
    def get_q(self) :
        return bg_annealer.get_q(self._ext, self.dtype)

    def randomize_q(self) :
        bg_annealer.randomize_q(self._ext, self.dtype)

    def calculate_E(self) :
        bg_annealer.calculate_E(self._ext, self.dtype)

    def init_anneal(self) :
        bg_annealer.init_anneal(self._ext, self.dtype)
        
    def anneal_one_step(self, G, kT) :
        bg_annealer.anneal_one_step(self._ext, G, kT, self.dtype)

    def fin_anneal(self) :
        bg_annealer.fin_anneal(self._ext, self.dtype)
        N0, N1, m = self.get_problem_size()
        self._E = np.empty((m), self.dtype)
        bg_annealer.get_E(self._ext, self._E, self.dtype)

        
def bipartite_graph_annealer(b0 = None, b1 = None, W = None, \
                             optimize = sqaod.minimize, n_trotters = None, \
                             dtype = np.float64) :
    return BipartiteGraphAnnealer(b0, b1, W, optimize, n_trotters, dtype)


if __name__ == '__main__' :
    N0 = 40
    N1 = 40
    m = 20
    
    np.random.seed(0)
            
    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5

    an = bipartite_graph_annealer(b0, b1, W, sqaod.minimize, m)
    
    Ginit = 5
    Gfin = 0.01
    kT = 0.02
    tau = 0.99
    n_repeat = 10

    for loop in range(0, n_repeat) :
        an.randomize_q()
        an.init_anneal()
        G = Ginit
        while Gfin < G :
            an.anneal_one_step(G, kT)
            G = G * tau
        an.fin_anneal()

        E = an.get_E()
        x = an.get_x()
        print E.shape, E
        print x
