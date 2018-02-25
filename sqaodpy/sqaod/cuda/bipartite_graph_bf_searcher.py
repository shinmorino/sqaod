import numpy as np
import random
import sys
import sqaod
from sqaod.common import checkers
import cuda_bg_bf_searcher as bg_bf_searcher
import device

class BipartiteGraphBFSearcher :
    
    def __init__(self, b0, b1, W, optimize, dtype) :
        self.dtype = dtype
        self._ext = bg_bf_searcher.new_searcher(dtype)
        self.assign_device(device.active_device)
        if not W is None :
            self.set_problem(b0, b1, W, optimize)
	self.ext_module = bg_bf_searcher
            
    def __del__(self) :
        self.ext_module.delete_searcher(self._ext, self.dtype)
	self.ext_module = None

    def assign_device(self, dev) :
        bg_bf_searcher.assign_device(self._ext, dev._ext, self.dtype)

    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)
        b0, b1, W = sqaod.clone_as_ndarray_from_vars([b0, b1, W], self.dtype)
        self._dim = (b0.shape[0], b1.shape[0])
        bg_bf_searcher.set_problem(self._ext, b0, b1, W, optimize, self.dtype)
        self._optimize = optimize

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return bg_bf_searcher.get_E(self._ext, self.dtype);

    def get_x(self) :
        return bg_bf_searcher.get_x(self._ext, self.dtype);
    
    def init_search(self) :
        bg_bf_searcher.init_search(self._ext, self.dtype);
        
    def fin_search(self) :
        bg_bf_searcher.fin_search(self._ext, self.dtype);
        
    def search_range(self, iBegin0, iEnd0, iBegin1, iEnd1) :
        bg_bf_searcher.search_range(self._ext, iBegin0, iEnd0, iBegin1, iEnd1, self.dtype)
        
    def search(self) :
        # one liner.  does not accept ctrl+c.
        bg_bf_searcher.search(self._ext, self.dtype);
        

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

        

def bipartite_graph_bf_searcher(b0 = None, b1 = None, W = None, optimize = sqaod.minimize, dtype = np.float64) :
    searcher = BipartiteGraphBFSearcher(b0, b1, W, optimize, dtype)
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
