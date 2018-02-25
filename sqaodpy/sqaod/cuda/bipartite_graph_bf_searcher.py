import numpy as np
import random
import sys
import sqaod
from sqaod.common import checkers
import cuda_bg_bf_searcher as cext
import device

class BipartiteGraphBFSearcher :

    _cext = cext
    _active_device = device.active_device
    
    def __init__(self, b0, b1, W, optimize, dtype, prefdict) :
        self.dtype = dtype
        self._cobj = cext.new_searcher(dtype)
        self.assign_device(device.active_device)
        if not W is None :
            self.set_problem(b0, b1, W, optimize)
        self.set_preferences(prefdict)
            
    def __del__(self) :
        cext.delete_searcher(self._cobj, self.dtype)

    def assign_device(self, dev) :
        cext.assign_device(self._cobj, dev._cobj, self.dtype)

    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)
        b0, b1, W = sqaod.clone_as_ndarray_from_vars([b0, b1, W], self.dtype)
        self._dim = (b0.shape[0], b1.shape[0])
        cext.set_problem(self._cobj, b0, b1, W, optimize, self.dtype)
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
        return cext.get_E(self._cobj, self.dtype);

    def get_x(self) :
        return cext.get_x(self._cobj, self.dtype);
    
    def init_search(self) :
        cext.init_search(self._cobj, self.dtype);
        
    def fin_search(self) :
        cext.fin_search(self._cobj, self.dtype);
        
    def search_range(self, iBegin0, iEnd0, iBegin1, iEnd1) :
        cext.search_range(self._cobj, iBegin0, iEnd0, iBegin1, iEnd1, self.dtype)
        
    def search(self) :
        # one liner.  does not accept ctrl+c.
        cext.search(self._cobj, self.dtype);
        

    def _search(self) :
        nStep = 1024
        self.init_search()

        N0, N1 = self._dim[0], self._dim[1]
        iMax = 1 << N0
        jMax = 1 << N1

        iStep = min(nStep, iMax)
        jStep = min(nStep, jMax)
        for iTile in range(0, iMax, iStep) :
            for jTile in range(0, jMax, jStep) :
                self.search_range(iTile, iTile + iStep, jTile, jTile + jStep)
        
        self.fin_search()

        

def bipartite_graph_bf_searcher(b0 = None, b1 = None, W = None,
                                optimize = sqaod.minimize, dtype = np.float64,
                                **prefs) :
    searcher = BipartiteGraphBFSearcher(b0, b1, W, optimize, dtype, prefs)
    return searcher


if __name__ == '__main__' :
    N0 = 14
    N1 = 5
    
    np.random.seed(0)

    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    
    bf = bipartite_graph_bf_searcher(b0, b1, W)
    bf.search()
    E = bf.get_E()
    x = bf.get_x() 
    print E.shape, E
    print x
