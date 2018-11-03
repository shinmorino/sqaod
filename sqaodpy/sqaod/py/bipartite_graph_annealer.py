from __future__ import print_function
from __future__ import division
import numpy as np
from types import MethodType
import sqaod
from sqaod import algorithm as algo
from sqaod.common import checkers
from . import formulas


class BipartiteGraphAnnealer :
    """ python implementation of bipartite graph annealer.

    This class is a reference implementation of bipartite graph annealers.
    All algorithms are written in python.
    """

    def __init__(self, b0, b1, W, optimize, prefdict) : # n_trotters
        if not W is None :
            self.set_qubo(b0, b1, W, optimize)
        self._select_algorithm(algo.coloring)    
        self.set_preferences(prefdict)

    def _vars(self) :
        return self._h0, self._h1, self._J, self._c, self._q0, self._q1

    def seed(self, seed) :
        """ set a seed value for random number generators.

        Args:
          seed (int or long) :
            seed value

        Note:
          sqaod.py annealer uses global random number genrator, so seed value is ignored.
          For annealers in other packages, given seed value is used as documented.  

        """
        # this annealer use global random number genertor.
        # nothing to do here.
        pass
    
    def get_problem_size(self) :
        """ get problem size.

        Problem size is defined as a number of bits of QUBO.

        Returns:
          tuple containing problem size, (N0, N1).
        """
        return self._N0, self._N1;
        
    def set_qubo(self, b0, b1, W, optimize = sqaod.minimize) :
        """ set QUBO.

        Args:
          numpy.ndarray b0, b1, W : QUBO.
          optimize : optimize direction, `sqaod.maximize, sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.
        """
        checkers.bipartite_graph.qubo(b0, b1, W)

        self._N0 = W.shape[1]
        self._N1 = W.shape[0]
        self._m = (self._N0 + self._N1) // 4
        self._optimize = optimize
        h0, h1, J, c = formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W)
        self._h0, self._h1 = optimize.sign(h0), optimize.sign(h1)
        self._J, self._c = optimize.sign(J), optimize.sign(c)

    def set_hamiltonian(self, h0, h1,J, c) :
        """ set Hamiltonian.

        Args:
          numpy.ndarray : h0(vector), h1(vector), J(matrix)
          floating point number : c

          Shapes for h0, h1, J must be (N0), (N1), (N1, N0) where N0 and N1 are number of bits.
        """
        checkers.bipartite_graph.hJc(h0, h1, J, c)

        self._N0 = J.shape[1]
        self._N1 = J.shape[0]
        self._m = (self._N0 + self._N1) // 4
        self._optimize = sqaod.minimize
        self._h0, self._h1, self._J, self._c = h0, h1, J, c
        
    def _select_algorithm(self, algoname) :
        if algoname == algo.coloring :
            self.anneal_one_step = \
                MethodType(BipartiteGraphAnnealer.anneal_one_step_coloring, self)
        elif algoname == algo.sa_naive :
            self.anneal_one_step = \
                MethodType(BipartiteGraphAnnealer.anneal_one_step_sa_naive, self)
        elif algoname == algo.sa_coloring :
            self.anneal_one_step = \
                MethodType(BipartiteGraphAnnealer.anneal_one_step_sa_coloring, self)
        else : # algoname == algo.naive :
            self.anneal_one_step = \
                MethodType(BipartiteGraphAnnealer.anneal_one_step_naive, self)

    def _get_algorithm(self) :
        if self.anneal_one_step.__func__ == BipartiteGraphAnnealer.anneal_one_step_naive :
            return algo.naive;
        elif self.anneal_one_step.__func__ == BipartiteGraphAnnealer.anneal_one_step_sa_naive :
            return algo.sa_naive;
        elif self.anneal_one_step.__func__ == BipartiteGraphAnnealer.anneal_one_step_sa_coloring :
            return algo.sa_coloring;
        return algo.coloring
            
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
          tuple of 2 numpy.int8 arrays : array of bit {0, 1}.

        x0.shape and x1.shape are (m, N0) and (m, N1) repsectively, and m is number of trotters.

        Note:
          calculate_E() or make_solution() should be called before calling get_E().
          ( CPU/CUDA annealers automatically/internally call calculate_E().)
        """
        return self._x_pairs

    def set_q(self, qpair) :
        self.prepare()
        for idx in range(0, self._m) :
            self._q0[idx] = qpair[0]
            self._q1[idx] = qpair[1]

    def set_qset(self, qpair) :
        qpairs = qpair
        self._m = len(qpairs);
        self.prepare()
        for idx in range(0, self._m) :
            self._q0[idx] = qpairs[idx][0]
            self._q1[idx] = qpairs[idx][1]

    # Ising model / spin
    
    def get_hamiltonian(self) :
        """ get hamiltonian.
        
        Returns: tuple of Hamiltonian variables
          h0(vector), h1(vector), J(matrix), c(scalar)
        """
        return self._h0, self._h1, self._J, self._c
            
    def get_q(self) :
        """ get spins.
        
        Returns: tuple for (q0, q1).  The q0 and q1 are numpy.int8 arrays whose lengths are N0 and N1 respectively.
        """
        qlist = []
        for qpair in zip(self._q0, self._q1) :
            qlist.append(qpair)
        return qlist

    def randomize_spin(self) :
        """ randomize spin. """
        sqaod.randomize_spin(self._q0)
        sqaod.randomize_spin(self._q1)

    def calculate_E(self) :
        """ calculate QUBO energy.
        
        This method calculate QUBO energy, and caches it, does not return any value.

        Note:
          A call to this method can be asynchronous.
        """
        h0, h1, J, c, q0, q1 = self._vars()
        E = np.empty((self._m), J.dtype)
        for idx in range(self._m) :
            # FIXME: 1d output for batch calculation
            E[idx] = formulas.bipartite_graph_calculate_E_from_spin(h0, h1, J, c, q0[idx], q1[idx])
        self._E = self._optimize.sign(E)

    def prepare(self) :
        """ preparation of internal resources.
        
        Note:
          prepare() should be called prior to run annealing loop.
        """
        self._q0 = np.empty((self._m, self._N0), dtype=np.int8)
        self._q1 = np.empty((self._m, self._N1), dtype=np.int8)
        if self._m == 1 :
            cur_algo = self._get_algorithm()
            if  cur_algo != algo.sa_naive and cur_algo != algo.sa_coloring :
                self._select_algorithm(algo.sa_naive)

    def make_solution(self) :
        """ make bit arrays(x) and calculate QUBO energy.

        Note:
          A call to this method can be asynchronous.
        """
        self._x_pairs = []
        for idx in range(self._m) :
            x0 = sqaod.bit_from_spin(self._q0[idx])
            x1 = sqaod.bit_from_spin(self._q1[idx])
            self._x_pairs.append((x0, x1))
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
        h0, h1, J, c, q0, q1 = self._vars()
        N0, N1 = self.get_problem_size()
        m = self._m
        N = N0 + N1
        twoDivM = 2. / m
        tempCoef = np.log(np.tanh(G * beta / m)) * beta

        for loop in range(N * m) :
            iq = np.random.randint(N)
            im = np.random.randint(m)
            mNeibour0 = (im + m - 1) % m
            mNeibour1 = (im + 1) % m
            
            if (iq < N0) :
                q = q0[im][iq]
                dE = twoDivM * q * (h0[iq] + np.dot(J.T[iq], q1[im]))
                dE -= q * (q0[mNeibour0][iq] + q0[mNeibour1][iq]) * tempCoef
                thresh = 1 if dE < 0 else np.exp(-dE * beta) 
                if thresh > np.random.rand():
                    q0[im][iq] = -q
            else :
                iq -= N0
                q = q1[im][iq]
                dE = twoDivM * q * (h1[iq] + np.dot(J[iq], q0[im]))
                dE -= q * (q1[mNeibour0][iq] + q1[mNeibour1][iq]) * tempCoef
                thresh = 1 if dE < 0 else np.exp(-dE * beta) 
                if thresh > np.random.rand():
                    q1[im][iq] = -q
        
    def _anneal_half_step_coloring(self, N, qAnneal, h, J, qFixed, G, beta, m) :
        """ (sqaod.py only) try to flip spins in a colored plane for SQA. """
        dEmat = np.matmul(J, qFixed.T)
        twoDivM = 2. / m
        tempCoef = np.log(np.tanh(G * beta / m)) * beta
        for im in range(0, m, 2) :
            mNeibour0 = (im + m - 1) % m
            mNeibour1 = (im + 1) % m
            for iq in range(0, N) :
                q = qAnneal[im][iq]
                dE = twoDivM * q * (h[iq] + dEmat[iq, im])
                dE -= q * (qAnneal[mNeibour0][iq] + qAnneal[mNeibour1][iq]) * tempCoef
                thresh = 1 if dE < 0 else np.exp(-dE * beta) 
                if thresh > np.random.rand():
                    qAnneal[im][iq] = -q
        for im in range(1, m, 2) :
            mNeibour0 = (im + m - 1) % m
            mNeibour1 = (im + 1) % m
            for iq in range(0, N) :
                q = qAnneal[im][iq]
                dE = twoDivM * q * (h[iq] + dEmat[iq, im])
                dE -= q * (qAnneal[mNeibour0][iq] + qAnneal[mNeibour1][iq]) * tempCoef
                thresh = 1 if dE < 0 else np.exp(-dE * beta) 
                if thresh > np.random.rand():
                    qAnneal[im][iq] = -q
                
    def anneal_one_step_coloring(self, G, beta) :
        """ (sqaod.py only) try to flip spins in a colored plane. """
        h0, h1, J, c, q0, q1 = self._vars()
        N0, N1 = self.get_problem_size()
        m = self._m
        self._anneal_half_step_coloring(N1, q1, h1, J, q0, G, beta, m)
        self._anneal_half_step_coloring(N0, q0, h0, J.T, q1, G, beta, m)

    # simulated annealing
    def anneal_one_step_sa_naive(self, kT, beta) :
        """ (sqaod.py only) sqaod.algorithm.sa_naive version of SA """
        h0, h1, J, c, q0, q1 = self._vars()
        N0, N1 = self.get_problem_size()
        m = self._m
        N = N0 + N1

        for im in range(m) :
            q0m, q1m = q0[im],q1[im]

            for loop in range(N) :
                iq = np.random.randint(N)
                if (iq < N0) :
                    q = q0m[iq]
                    dE = 2. * q * (h0[iq] + np.dot(J.T[iq], q1m))
                    thresh = 1 if dE < 0 else np.exp(-dE * kT * beta) 
                    if thresh > np.random.rand():
                        q0m[iq] = -q
                else :
                    iq -= N0
                    q = q1m[iq]
                    dE = 2. * q * (h1[iq] + np.dot(J[iq], q0m))
                    thresh = 1 if dE < 0 else np.exp(-dE * kT * beta) 
                    if thresh > np.random.rand():
                        q1m[iq] = -q
                        
    def _anneal_half_step_sa_coloring(self, N, qAnneal, h, J, qFixed, kT, beta, m) :
        """ (sqaod.py only) try to flip spins in a colored plane for SA. """
        dEmat = np.matmul(J, qFixed.T)
        for im in range(m) :
            qAnnealm = qAnneal[im]
            for iq in range(0, N) :
                q = qAnnealm[iq]
                dE = 2. * q * (h[iq] + dEmat[iq, im])
                thresh = 1 if dE < 0 else np.exp(-dE * kT * beta) 
                if thresh > np.random.rand():
                    qAnnealm[iq] = -q
                
    def anneal_one_step_sa_coloring(self, kT, beta) :
        """ (sqaod.py only) sqaod.algorithm.sa_coloring version of SA """
        h0, h1, J, c, q0, q1 = self._vars()
        N0, N1 = self.get_problem_size()
        m = self._m
        self._anneal_half_step_sa_coloring(N1, q1, h1, J, q0, kT, beta, m)
        self._anneal_half_step_sa_coloring(N0, q0, h0, J.T, q1, kT, beta, m)
        


def bipartite_graph_annealer(b0 = None, b1 = None, W = None, \
                             optimize = sqaod.minimize, **prefs) :
    """ factory function for sqaod.py.BipartiteGraphAnnealer.

    Args:
      numpy.ndarray b0, b1, W : QUBO
      optimize : specify optimize direction, `sqaod.maximize or sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.
      prefs : `preference <preference.html>`_ as \*\*kwargs
    Returns:
      sqaod.py.BipartiteGraphAnnealer: annealer instance
    """
    return BipartiteGraphAnnealer(b0, b1, W, optimize, prefs)


if __name__ == '__main__' :
    N0 = 12
    N1 = 10
    m = 10
    
    np.random.seed(0)
            
    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5

    an = bipartite_graph_annealer(b0, b1, W, sqaod.minimize, n_trotters=m, algorithm=algo.coloring)
    
    Ginit = 5
    Gfin = 0.01
    beta = 1. / 0.02
    tau = 0.99
    n_repeat = 10

    for loop in range(0, n_repeat) :
        an.prepare()
        an.randomize_spin()
        G = Ginit
        while Gfin < G :
            an.anneal_one_step(G, beta)
            G = G * tau
        an.make_solution()
        
        E = an.get_E()
        x = an.get_x()
        print(E)
        #print(x)
