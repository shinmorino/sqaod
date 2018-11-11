import wildqat
import numpy as np
import sqaod

class opt(wildqat.opt) :
    def __init__(self, pkg = None, dtype = np.float32) :
        wildqat.opt.__init__(self)
        if pkg is None :
            pkg = sqaod.cpu
        self.ann = pkg.dense_graph_annealer(dtype = dtype)

    def sa(self):
        """
	Run SA with provided QUBO. 
	Set qubo attribute in advance of calling this method.
	"""
        self.E = []
        
        T = self.Ts
        if self.qubo != []:
            W = np.array(self.qubo)
            self.ann.set_qubo(W, sqaod.minimize)
            self.ann.set_preferences(algorithm = sqaod.algorithm.sa_naive, n_trotters = 1)

        self.ann.prepare()
        self.ann.randomize_spin()
        N = self.ann.get_problem_size()
        n_iters_at_T = (self.ite + N - 1) // N
        while T > self.Tf:
            for _ in range(n_iters_at_T) :
                self.ann.anneal_one_step(T, 1.)
                E = self.ann.get_system_E(0., 0.) # parameters are ignored.
                self.E.append(E)
            T *= self.R

        x = self.ann.get_x()

        return x[0]

    def sqa(self):
        """
        Run SQA with provided QUBO.
        Set qubo attribute in advance of calling this method.
        """
        self.E = []
        
        G = self.Gs
        if self.qubo != []:
            self.ann.set_qubo(self.qubo, sqaod.minimize)
            self.ann.set_preferences(algorithm = sqaod.algorithm.default, n_trotters = self.tro)
            self.qi()

        self.ann.prepare()
        self.ann.randomize_spin()
        N = self.ann.get_problem_size()
        n_flips_per_call = N * self.tro
        n_iters_at_G = (self.ite + n_flips_per_call - 1) // n_flips_per_call
        while G > self.Gf:
            for _ in range(n_iters_at_G) :
                self.ann.anneal_one_step(G, 1. / self.Tf)
            E = self.ann.get_system_E(G, 1. / self.Tf)
            self.E.append(E)
            G *= self.R

        x = self.ann.get_x()

        return x
