import numpy as np
import sqaod
from sqaod.common import checkers
import cpu_dg_bf_searcher as cext

class DenseGraphBFSearcher :

    _cext = cext
    
    def __init__(self, W, optimize, dtype, prefdict) :
        self.dtype = dtype
        self._cobj = cext.new_bf_searcher(dtype)
        if not W is None :
            self.set_problem(W, optimize)
        self.set_preferences(prefdict)
            
    def __del__(self) :
        cext.delete_bf_searcher(self._cobj, self.dtype)

    def set_problem(self, W, optimize = sqaod.minimize) :
        checkers.dense_graph.qubo(W)
        W = sqaod.clone_as_ndarray(W, self.dtype)
        self._N = W.shape[0]
        cext.set_problem(self._cobj, W, optimize, self.dtype)
        self._optimize = optimize

    def set_preferences(self, prefdict = None, **prefs) :
        if not prefdict is None :
            cext.set_preferences(self._cobj, prefdict, self.dtype)
        cext.set_preferences(self._cobj, prefs, self.dtype)

    def get_preferences(self) :
        return cext.get_preferences(self._cobj, self.dtype);

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return cext.get_E(self._cobj, self.dtype)

    def get_x(self) :
        return cext.get_x(self._cobj, self.dtype)

    def init_search(self) :
        cext.init_search(self._cobj, self.dtype);
        
    def fin_search(self) :
        cext.fin_search(self._cobj, self.dtype);
        
    def search_range(self, iBegin0, iEnd0, iBegin1, iEnd1) :
        cext.search_range(self._cobj, iBegin0, iEnd0, iBegin1, iEnd1, self.dtype)
        
    def search(self) :
        N = self._N
        iMax = 1 << N
        iStep = min(512, iMax)
        self.init_search()
        for iTile in range(0, iMax, iStep) :
            cext.search_range(self._cobj, iTile, iTile + iStep, self.dtype)
        self.fin_search()
        
    def _search(self) :
        # one liner.  does not accept ctrl+c.
        cext.search(self._cobj, self.dtype)


def dense_graph_bf_searcher(W = None, optimize = sqaod.minimize, dtype=np.float64, **prefs) :
    return DenseGraphBFSearcher(W, optimize, dtype, prefs)


if __name__ == '__main__' :

    np.random.seed(0)
    dtype = np.float32
    N = 8
    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])

    N = 16
    W = sqaod.generate_random_symmetric_W(N, -0.5, 0.5, dtype);
    bf = dense_graph_bf_searcher(W, sqaod.minimize, dtype)
    bf.set_preferences(tile_size=256)
    
    import time

    start = time.time()
    bf.search()
    end = time.time()
    print(end - start)
    
    x = bf.get_x() 
    E = bf.get_E()
    print E
    print x

#    import sqaod.py.formulas
#    E = np.empty((1), dtype)
#    for bits in x :
#        print sqaod.py.formulas.dense_graph_calculate_E(W, bits)
