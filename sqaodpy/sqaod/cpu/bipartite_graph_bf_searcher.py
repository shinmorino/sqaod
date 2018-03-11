import numpy as np
import random
import sys
import sqaod
from sqaod.common import checkers
import cpu_bg_bf_searcher as cext


class BipartiteGraphBFSearcher :

    _cext = cext
    
    def __init__(self, b0, b1, W, optimize, dtype, prefdict) :
        self.dtype = dtype
        self._optimize = optimize
        self._cobj = cext.new_searcher(dtype)
        if not W is None :
            self.set_qubo(b0, b1, W, optimize)
        self.set_preferences(prefdict)
            
    def __del__(self) :
        cext.delete_searcher(self._cobj, self.dtype)

    def set_qubo(self, b0, b1, W, optimize = sqaod.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)
        b0, b1, W = sqaod.clone_as_ndarray_from_vars([b0, b1, W], self.dtype)
        self._dim = (b0.shape[0], b1.shape[0])
        cext.set_qubo(self._cobj, b0, b1, W, optimize, self.dtype)

    def get_problem_size(self) :
        return cext.get_problem_size(self._cobj, self.dtype)

    def set_preferences(self, prefdict=None, **prefs) :
        if not prefdict is None :
            cext.set_preferences(self._cobj, prefdict, self.dtype)
        cext.set_preferences(self._cobj, prefs, self.dtype)

    def get_preferences(self) :
        return cext.get_preferences(self._cobj, self.dtype);
        
    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return cext.get_E(self._cobj, self.dtype);

    def get_x(self) :
        return cext.get_x(self._cobj, self.dtype);
    
    def prepare(self) :
        cext.prepare(self._cobj, self.dtype);
        
    def make_solution(self) :
        cext.make_solution(self._cobj, self.dtype);
        
    def search_range(self) :
        return cext.search_range(self._cobj, self.dtype)
        
    def _search(self) :
        # one liner.  does not accept ctrl+c.
        cext.search(self._cobj, self.dtype);

    def search(self) :
        self.prepare()
        while True :
            comp, curx0, curx1 = cext.search_range(self._cobj, self.dtype)
            if comp :
                break;
        self.make_solution()
        

def bipartite_graph_bf_searcher(b0 = None, b1 = None, W = None, optimize = sqaod.minimize, dtype = np.float64, **prefs) :
    return BipartiteGraphBFSearcher(b0, b1, W, optimize, dtype, prefs)


if __name__ == '__main__' :
    N0 = 14
    N1 = 5
    
    np.random.seed(0)

    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    
    bf = bipartite_graph_bf_searcher(b0, b1, W)
    bf.set_preferences(tile_size_0=256, tile_size_1=256)
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print E.shape, E
    print x
    print bf.get_preferences()
