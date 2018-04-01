from __future__ import print_function
import numpy as np
import sqaod
from sqaod.common.bipartite_graph_bf_searcher_base import BipartiteGraphBFSearcherBase
from . import cuda_bg_bf_searcher as cext
from . import device

class BipartiteGraphBFSearcher(BipartiteGraphBFSearcherBase) :
    
    def __init__(self, b0, b1, W, optimize, dtype, prefdict) :
        self._cobj = cext.new(dtype)
        cext.assign_device(self._cobj, device.active_device._cobj, dtype)
        self._device = device.active_device
        BipartiteGraphBFSearcherBase.__init__(self, cext, dtype, b0, b1, W, optimize, prefdict)


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
    print(E.shape, E)
    print(x)
