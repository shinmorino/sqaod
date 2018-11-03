from __future__ import print_function
import numpy as np
import sys
import sqaod
from sqaod import algorithm as algo
from . import formulas
from sqaod.common import checkers, symmetrize

class DenseGraphBFSearcher :
    """ Dense graph brute-force searcher"""

    def __init__(self, W, optimize, prefdict) :
        if W is not None :
            self.set_qubo(W, optimize)
        self._tile_size = 1024
        self.set_preferences(prefdict)
            
    def set_qubo(self, W, optimize = sqaod.minimize) :
        """ set QUBO.

        Args:
          numpy.ndarray W :
            QUBO matrix.  W should be a sqaure matrix.
            Upper/Lower triangular matrices or symmetric matrices are accepted.
          optimize : optimize direction, `sqaod.maximize, sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`        """
        checkers.dense_graph.qubo(W)
        W = symmetrize(W)
        N = W.shape[0]
        self._W = optimize.sign(W)
        self._N = N
        self._x = []
        self._optimize = optimize

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
        v = prefdict.get('tile_size')
        if v is not None :
            self._tile_size = v;
    
    def get_problem_size(self) :
        """ get problem size.

        Problem size is defined as a number of bits of QUBO.

        Returns:
          tuple containing problem size, (N0, N1).
        """
        return self._N
            
    def get_preferences(self) :
        """ get solver preferences.

        Returns:
          dict: preference dictionary.

        References:
          `preference <preference.html>`_
        """
        prefs = {}
        prefs['tile_size'] = self._tile_size
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
        return self._x

    def prepare(self) :
        """ preparation of internal resources.
        
        Note:
          prepare() should be called prior to run annealing loop.
        """
        
        N = self._N
        self._tile_size = min(1 << N, self._tile_size)
        self._xMax = 1 << N
        self._Emin = sys.float_info.max
        self._xbegin = 0

    def make_solution(self) :
        """ calculate QUBO energy.
        
        This method calculate QUBO energy, and caches it, does not return any value.

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
        nMinX = len(self._x)
        self._E = np.empty((nMinX))
        self._E[...] = self._optimize.sign(self._Emin)
        
    def search_range(self) :
        """ Search minimum of QUBO energy within a range.

        Returns:
          True if search completed, False otherwise.

        This method enables stepwise minimum QUBO energy search.
        One method call covers a range specified by 'tile_size'.
        When a given problem is big, whole search takes very long time.
        By using this method, you can do searches stepwise.
        """
        
        xBegin = self._xbegin
        xEnd = self._tile_size
        
        N = self._N
        W = self._W
        xBegin = max(0, min(self._xMax, xBegin))
        xEnd = max(0, min(self._xMax, xEnd))
        x = sqaod.create_bitset_sequence(range(xBegin, xEnd), N)
        Etmp = formulas.dense_graph_batch_calculate_E(W, x)
        for i in range(xEnd - xBegin) :
            if self._Emin < Etmp[i] :
                continue
            elif Etmp[i] < self._Emin :
                self._Emin = Etmp[i]
                self._x = [x[i]]
            else :
                self._x.append(x[i])

        self._xbegin = xEnd
        return self._xbegin == self._xMax

    def search(self) :
        """ One liner for brute-force search. """
        self.prepare()
        xStep = min(self._tile_size, self._xMax)
        while not self.search_range() :
            pass
        self.make_solution()
        
    
def dense_graph_bf_searcher(W = None, optimize = sqaod.minimize, **prefs) :
    """ factory function for sqaod.py.DenseGraphAnnealer.

    Args:
      numpy.ndarray : QUBO
      optimize : specify optimize direction, `sqaod.maximize or sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.
      prefs : `preference <preference.html>`_ as \*\*kwargs
    Returns:
      sqaod.py.DenseGraphBFSearcher: brute-force searcher instance
    """
    
    return DenseGraphBFSearcher(W, optimize, prefs)


if __name__ == '__main__' :

    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    
    
    N = 8
    bf = dense_graph_bf_searcher(W, sqaod.minimize)
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print(E, len(x), x[0])

    bf = dense_graph_bf_searcher(W, sqaod.maximize)
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print(E, len(x), x[0])
