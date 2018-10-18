from __future__ import print_function
import numpy as np
import sys
import sqaod
from sqaod import algorithm as algo
from . import formulas
from sqaod.common import checkers, symmetrize

class DenseGraphBFSearcher :
    
    def __init__(self, W, optimize, prefdict) :
        if W is not None :
            self.set_qubo(W, optimize)
        self._tile_size = 1024
        self.set_preferences(prefdict)
            
    def set_qubo(self, W, optimize = sqaod.minimize) :
        checkers.dense_graph.qubo(W)
        W = symmetrize(W)
        N = W.shape[0]
        self._W = optimize.sign(W)
        self._N = N
        self._x = []
        self._optimize = optimize

    def select_algorithm(self, algoname) :
        pass # always choose brute-force search.

    def _get_algorithm(self) :
        return algo.brute_force_search
            
    def set_preferences(self, prefdict=None, **prefs) :
        if not prefdict is None :
            self._set_prefdict(prefdict)
        self._set_prefdict(prefs)
        
    def _set_prefdict(self, prefdict) :
        v = prefdict.get('tile_size')
        if v is not None :
            self._tile_size = v;
    
    def get_problem_size(self) :
        return self._N
            
    def get_preferences(self) :
        prefs = {}
        prefs['tile_size'] = self._tile_size
        prefs['algorithm'] = self._get_algorithm()
        return prefs

    def get_optimize_dir(self) :
        return self._optimize
    
    def get_E(self) :
        return self._E
        
    def get_x(self) :
        return self._x

    def prepare(self) :
        N = self._N
        self._tile_size = min(1 << N, self._tile_size)
        self._xMax = 1 << N
        self._Emin = sys.float_info.max
        self._xbegin = 0

    def make_solution(self) :
        self.calculate_E()

    def calculate_E(self) :
        nMinX = len(self._x)
        self._E = np.empty((nMinX))
        self._E[...] = self._optimize.sign(self._Emin)
        
    def search_range(self) :
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
        self.prepare()
        xStep = min(self._tile_size, self._xMax)
        while not self.search_range() :
            pass
        self.make_solution()
        
    
def dense_graph_bf_searcher(W = None, optimize = sqaod.minimize, **prefs) :
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
