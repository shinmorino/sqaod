from __future__ import print_function
from __future__ import division
import numpy as np
import sqaod
from . import formulas
from sqaod.common import checkers, symmetrize
from types import MethodType
from sqaod import algorithm as algo

class DenseGraphAnnealer :
    """ python implementation of dense graph annealer.

    This class is a reference implementation of dense graph annealers.
    All algorithms are written in python.
    """
    def __init__(self, W, optimize, prefdict) :
        if not W is None :
            self.set_qubo(W, optimize)
        self._select_algorithm(algo.coloring)
        self.set_preferences(prefdict)

    def seed(self, seed) :
        """ set a seed value for random number generators.

        Args:
          seed (int or long) :
            seed value

        Note:
          sqaod.py annealer uses global random number genrator, so seed value is ignored.
          For annealers in other packages, given seed value is used as documented.  

        """
        # py version uses using global random generator.
        # Nothing to do here.
        pass
        
    def _vars(self) :
        return self._h, self._J, self._c, self._q
    
    def get_problem_size(self) :
        """ get problem size.

        Problem size is defined as a number of bits of QUBO.

        Returns:
          int: problem size.
        """
        return self._N;

    def set_qubo(self, W, optimize = sqaod.minimize) :
        """ set QUBO.

        Args:
          numpy.ndarray W :
            QUBO matrix.  W should be a sqaure matrix.
            Upper/Lower triangular matrices or symmetric matrices are accepted.
          optimize : optimize direction, `sqaod.maximize, sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`        """
        
        checkers.dense_graph.qubo(W)
        W = symmetrize(W)

        h, J, c = formulas.dense_graph_calculate_hamiltonian(W)
        self._optimize = optimize
        self._h, self._J, self._c = optimize.sign(h), optimize.sign(J), optimize.sign(c)
        self._N = W.shape[0]
        self._m = self._N // 2
        
    def set_hamiltonian(self, h, J, c) :
        """ set Hamiltonian.

        Args:
          numpy.ndarray : h(vector), J(matrix)
          floating point number : c

          Shapes for h, J must be (N), (N, N) where N is problem size.

          J must be a upper/lower triangular or symmetric matrix.
        """
        
        checkers.dense_graph.hJc(h, J, c)
        J = symmetrize(J)
        self._optimize = sqaod.minimize
        self._h, self._J, self._c = h, J, c
        self._N = J.shape[0]
        self._m = self._N // 2
        
    def _select_algorithm(self, algoname) :
        if algoname == algo.coloring :
            self.anneal_one_step = \
                MethodType(DenseGraphAnnealer.anneal_one_step_coloring, self)
        elif algoname == algo.sa_naive :
            self.anneal_one_step = \
                MethodType(DenseGraphAnnealer.anneal_one_step_sa_naive, self)
        else :
            self.anneal_one_step = \
                MethodType(DenseGraphAnnealer.anneal_one_step_naive, self)

    def _get_algorithm(self) :
        if self.anneal_one_step.__func__ == DenseGraphAnnealer.anneal_one_step_coloring :
            return algo.coloring;
        if self.anneal_one_step.__func__ == DenseGraphAnnealer.anneal_one_step_sa_naive :
            return algo.sa_naive;
        return algo.naive
            
    def set_preferences(self, prefdict = None, **prefs) :
        """ set solver preferences.

        Args:
          prefdict(dict) : preference dictionary.
          prefs(dict) : preference dictionary as \*\*kwargs. 

        References:
          `preference <preference.html>`_
        """
        if not prefdict is None :
            self._set_prefdict(prefdict)
        self._set_prefdict(prefs)
        
    def _set_prefdict(self, prefdict) :
        v = prefdict.get('n_trotters')
        if v is not None :
            self._m = v;
        v = prefdict.get('algorithm')
        if v is not None :
            self._select_algorithm(v)

    def get_preferences(self) :
        """ get solver preferences.

        Returns:
          dict: preference dictionary.

        References:
          `preference <preference.html>`_
        """
        prefs = { }
        if hasattr(self, '_m') :
            prefs['n_trotters'] = self._m
        prefs['algorithm'] = self._get_algorithm()
        return prefs
        
    def get_optimize_dir(self) :
        """ get optimize direction
        
        Returns:
          optimize direction, `sqaod.maximize, sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.

        """
        return self._optimize

    def get_E(self) :
        """ get QUBO energy.

        Returns:
          array of floating point number : QUBO energy.

        QUBO energy value is calculated for each trotter. so E is a vector whose length is m.

        Notes:
          You need to call calculate_E() or make_solution() before calling get_E().
          CPU/CUDA versions of solvers automatically call calculate_E() in get_E()
        """
        return self._E

    def get_x(self) :
        """ get bits.

        Returns:
          numpy.int8 : array of bit {0, 1}.

        x.shape is (m, N) where N is problem size, and m is number of trotters.

        Note:
          calculate_E() or make_solution() should be called before calling get_E().
          ( CPU/CUDA annealers automatically/internally call calculate_E().)
        """
        return self._x

    def set_q(self, q) :
        """ set spins.

        Args:
          q(np.array of np.int8) : array (length = N) of spin {-1, 1}.
        """

        if q.dtype != np.int8 :
            q = np.asarray(q, np.int8)
        self._q[:] = q

    def set_qset(self, q) :
        """ set "array of" spins.
        
        Args:
          qset (list of np.int8 array ) : list of array of spin {-1, 1}.


        qset is a list (length = m) of int8 array (length = N).
        """
        
        self._m = len(q)
        self.prepare()
        qlist = q
        for idx in range(len(qlist)) :
            q = qlist[idx]
            if q.dtype != np.int8 :
                q = np.asarray(q, np.int8)
            self._q[idx] = q
    
    # Ising model

    def get_hamiltonian(self) :
        """ get hamiltonian.
        
        Returns: tuple of Hamiltonian variables
          h(vector), J(symmeric matrix), c(scalar)
        """
        return self._h, self._J, self._c

    def get_q(self) :
        """ get spins.
        
        Returns:
          numpy.int8 : array of spin {-1, 1}.
        """
        return self._q
    
    def randomize_spin(self) :
        """ randomize spin. """
        sqaod.randomize_spin(self._q)

    def calculate_E(self) :
        """ calculate QUBO energy.
        
        This method calculate QUBO energy, and caches it, does not return any value.

        Note:
          A call to this method can be asynchronous.
        """
        
        h, J, c, q = self._vars()
        E = formulas.dense_graph_batch_calculate_E_from_spin(h, J, c, q)
        self._E = self._optimize.sign(E)

    def prepare(self) :
        """ preparation of internal resources.
        
        Note:
          prepare() should be called prior to run annealing loop.
        """
        if self._m == 1 :
            self._select_algorithm(algo.sa_naive)
        self._q = np.empty((self._m, self._N), dtype=np.int8)

    def make_solution(self) :
        """ make bit arrays(x) and calculate QUBO energy.

        Note:
          A call to this method can be asynchronous.
        """
        self._x = []
        for idx in range(self._m) :
            x = sqaod.bit_from_spin(self._q[idx])
            self._x.append(x)
        self.calculate_E()

    def anneal_one_step(self, G, beta) :
        """ Run annealing one step.

        Args:
          G (floating point number) : G in SQA, kT in SA.
          beta (floating point number) : inverse temperature.
        """

        # will be dynamically replaced.
        pass
        
    def anneal_one_step_naive(self, G, beta) :
        """ (sqaod.py only) sqaod.algorithm.naive version of SQA """
        
        h, J, c, q = self._vars()
        N = self._N
        m = self._m
        two_div_m = 2. / np.float64(m)
        coef = np.log(np.tanh(G * beta / m)) * beta
        
        for i in range(self._N * self._m):
            x = np.random.randint(N)
            y = np.random.randint(m)
            qyx = q[y][x]
            sum = np.dot(J[x], q[y]); # diagnoal elements in J are zero.
            dE = two_div_m * qyx * (h[x] + sum)
            dE -= qyx * (q[(m + y - 1) % m][x] + q[(y + 1) % m][x]) * coef
            threshold = 1. if (dE <= 0.) else np.exp(-dE * beta)
            if threshold > np.random.rand():
                q[y][x] = - qyx

    def anneal_colored_plane(self, G, beta, offset) :
        """ (sqaod.py only) try to flip spins in a colored plane. """
        h, J, c, q = self._vars()
        N = self._N
        m = self._m
        two_div_m = 2. / np.float64(m)
        coef = np.log(np.tanh(G * beta / m)) * beta
        
        for y in range(self._m):
            x = (offset + np.random.randint(1 << 30) * 2) % N
            qyx = q[y][x]
            sum = np.dot(J[x], q[y]); # diagnoal elements in J are zero.
            dE = two_div_m * qyx * (h[x] + sum)
            dE -= qyx * (q[(m + y - 1) % m][x] + q[(y + 1) % m][x]) * coef
            threshold = 1. if (dE <= 0.) else np.exp(-dE * beta)
            if threshold > np.random.rand():
                q[y][x] = - qyx
            
    def anneal_one_step_coloring(self, G, beta) :
        """ (sqaod.py only) sqaod.algorithm.coloring version of SQA """
        for loop in range(0, self._N) :
            self.anneal_colored_plane(G, beta, 0)
            self.anneal_colored_plane(G, beta, 1)
    
    def anneal_one_step_sa_naive(self, kT, beta) :
        """ (sqaod.py only) sqaod.algorithm.sa_naive version of SA """
        h, J, c, q = self._vars()
        N = self._N

        for iq in range(self._m) :
            qm = q[iq]
            for i in range(self._N):
                x = np.random.randint(N)
                qx = qm[x]
                sum = np.dot(J[x], qm); # diagnoal elements in J are zero.
                dE = 2. * qx * (h[x] + sum)
                threshold = 1. if (dE <= 0.) else np.exp(-dE * kT * beta)
                if threshold > np.random.rand():
                    qm[x] = - qx

                
def dense_graph_annealer(W = None, optimize = sqaod.minimize, **prefs) :
    """ factory function for sqaod.py.DenseGraphAnnealer.

    Args:
      numpy.ndarray : QUBO
      optimize : specify optimize direction, `sqaod.maximize or sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.
      prefs : `preference <preference.html>`_ as \*\*kwargs
    Returns:
      sqaod.py.DenseGraphAnnealer: annealer instance
    """
    an = DenseGraphAnnealer(W, optimize, prefs)
    return an


if __name__ == '__main__' :

    np.random.seed(0)
    Ginit = 5.
    Gfin = 0.01
    
    nRepeat = 4
    beta = 1. / 0.02
    tau = 0.99
    
    N = 8
    m = 4
    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    

    #N = 20
    #m = 10
    #W = sqaod.generate_random_symmetric_W(N, -0.5, 0.5, np.float64)
    
    algoname = algo.default
    # algo = DenseGraphAnnealer.naive
    ann = dense_graph_annealer(W, sqaod.minimize, n_trotters=m)
    ann.set_preferences(algorithm = algo.naive)
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.prepare()
        ann.randomize_spin()
        while Gfin < G :
            ann.anneal_one_step(G, beta)
            G = G * tau

        ann.make_solution()
        E = ann.get_E()
        #x = ann.get_x()
        print(E) # ,x

    prefs = ann.get_preferences()
    print(prefs)
        
    ann = dense_graph_annealer(W, sqaod.maximize)
    ann.set_preferences(prefs)
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.prepare()
        ann.randomize_spin()
        while Gfin < G :
            ann.anneal_one_step(G, beta)
            G = G * tau

        ann.make_solution()
        E = ann.get_E()
        #x = ann.get_x()
        print(E) # ,x

    prefs = ann.get_preferences()
    print(prefs)
