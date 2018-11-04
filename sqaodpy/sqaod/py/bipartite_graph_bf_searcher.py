import numpy as np
import sys
import sqaod
from sqaod.common import checkers
from sqaod import algorithm as algo

class BipartiteGraphBFSearcher :
    """ Bipartite graph  brute-force searcher"""

    _tile_size_0_default = 512
    _tile_size_1_default = 512
    
    def __init__(self, W, b0, b1, optimize, prefdict) :
        self._verbose = False
        if not W is None :
            self.set_qubo(W, b0, b1, optimize)
        self._set_prefdict(prefdict)

    def _vars(self) :
        return self._b0, self._b1, self._W

    def get_problem_size(self) :
        """ get problem size.

        Problem size is defined as a number of bits of QUBO.

        Returns:
          tuple containing problem size, (N0, N1).
        """
        return self._N0, self._N1
    
    def set_qubo(self, b0, b1, W, optimize = sqaod.minimize) :
        """ set QUBO.

        Args:
          numpy.ndarray b0, b1, W : QUBO.
          optimize : optimize direction, `sqaod.maximize, sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.
        checkers.bipartite_graph.qubo(b0, b1, W)
        """
        self._N0 = b0.shape[0]
        self._N1 = b1.shape[0]
        self._optimize = optimize
        self._b0 = self._optimize.sign(b0)
        self._b1 = self._optimize.sign(b1)
        self._W = self._optimize.sign(W)
        self._tile_size_0 = BipartiteGraphBFSearcher._tile_size_0_default
        self._tile_size_1 = BipartiteGraphBFSearcher._tile_size_1_default

    def _get_algorithm(self) :
        return algo.brute_force_search

    def set_preferences(self, prefdict=None, **prefs) :
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
        v = prefdict.get('tile_size_0')
        if v is not None :
            self._tile_size_0 = v;
        v = prefdict.get('tile_size_1')
        if v is not None :
            self._tile_size_1 = v;

    def get_preferences(self) :
        """ get solver preferences.

        Returns:
          dict: preference dictionary.

        References:
          `preference <preference.html>`_
        """
        prefs = {}
        prefs['tile_size_0'] = self._tile_size_0
        prefs['tile_size_1'] = self._tile_size_1
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
        return self._xPairs

    def prepare(self) :
        """ preparation of internal resources.
        
        Note:
          prepare() should be called prior to run annealing loop.
        """
        N0, N1 = self.get_problem_size()
        self._x0begin = 0
        self._x1begin = 0
        self._x0max = 1 << N0
        self._x1max = 1 << N1
        self._tile_size_0 = min(self._x0max, self._tile_size_0)
        self._tile_size_1 = min(self._x1max, self._tile_size_1)
        self._Emin = sys.float_info.max
        self._xPairs = []

    def make_solution(self) :
        """ make bit arrays(x) and calculate QUBO energy.

        Note:
          A call to this method can be asynchronous.
        """
        self.calculate_E()

    def calculate_E(self) :
        """ calculate QUBO energy.
        
        This method calculate QUBO energy, and caches it, does not return any value.

        Note:
          A call to this method can be asynchronous.
        """
        nXmin = len(self._xPairs)
        self._E = np.empty((nXmin))
        self._E[...] = self._optimize.sign(self._Emin)

    # Not used.  Keeping it for reference.    
    def _search_naive(self) :
        N0, N1 = self._get_dim()
        b0, b1, W = self._vars()

        for i0 in range(self._x0max) :
            x0 = sqaod.create_bitset_sequence((i0), N0)
            for i1 in range(self._x1max) :
                x1 = sqaod.create_bitset_sequence((i1), N1)
                Etmp = np.dot(b0, x0.transpose()) + np.dot(b1, x1.transpose()) \
                       + np.dot(x1, np.matmul(W, x0.transpose()))
                if self._Emin < Etmp :
                    continue
                elif Etmp < self._Emin :
                    self._Emin = Etmp
                    self._xPairs = [(x0, x1)]
                else :
                    self._xPairs.append = [(x0, x1)]
        
    def search_range(self) :
        """ Search minimum of QUBO energy within a range.

        Returns:
          True if search completed, False otherwise.

        This method enables stepwise minimum QUBO energy search.
        One method call covers a range specified by 'tile_size'.
        When a given problem is big, whole search takes very long time.
        By using this method, you can do searches stepwise.
        """
        N0, N1 = self.get_problem_size()
        b0, b1, W = self._vars()
        x0begin = max(0, min(self._x0max, self._x0begin))
        x0end = max(0, min(self._x0max, x0begin + self._tile_size_0))
        x1begin = max(0, min(self._x1max, self._x1begin))
        x1end = max(0, min(self._x1max, x1begin + self._tile_size_1))

        x0step = x0end - x0begin
        x1step = x1end - x1begin

        bits0 = sqaod.create_bitset_sequence(range(x0begin, x0end), N0)
        bits1 = sqaod.create_bitset_sequence(range(x1begin, x1end), N1)
        Etmp = np.matmul(b0, bits0.T).reshape(1, x0step) \
               + np.matmul(b1, bits1.T).reshape(x1step, 1) + np.matmul(bits1, np.matmul(W, bits0.T))
        for x1 in range(x1step) :
            for x0 in range(x0step) :
                if self._Emin < Etmp[x1][x0] :
                    continue
                elif Etmp[x1][x0] < self._Emin :
                    self._Emin = Etmp[x1][x0]
                    self._xPairs = [(bits0[x0], bits1[x1])]
                else :
                    self._xPairs.append((bits0[x0], bits1[x1]))

        self._x1begin = x1end
        if self._x1begin != self._x1max :
            self._x1begin = x1end
            return False
        
        if self._x0begin == self._x0max :
            return True
        
        self._x1begin = 0
        self._x0begin = x0end
        return False

                    
    def search(self) :
        """ One liner for brute-force search. """
        self.prepare()

        while not self.search_range() :
            pass

        self.make_solution()
        
def bipartite_graph_bf_searcher(b0 = None, b1 = None, W = None, optimize = sqaod.minimize, **prefs) :
    """ factory function for sqaod.py.BipartiteGraphAnnealer.

    Args:
      numpy.ndarray b0, b1, W : QUBO
      optimize : specify optimize direction, `sqaod.maximize or sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.
      prefs : `preference <preference.html>`_ as \*\*kwargs
    Returns:
      sqaod.py.BipartiteGraphBFSearcher: annealer instance
    """
    return BipartiteGraphBFSearcher(b0, b1, W, optimize, prefs)


if __name__ == '__main__' :
    N0 = 14
    N1 = 9
    
    np.random.seed(0)

    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    
    bf = bipartite_graph_bf_searcher(b0, b1, W)
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print(E)
    print(x)
    print(bf.get_preferences())
