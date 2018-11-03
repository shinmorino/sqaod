from __future__ import print_function
import numpy as np
import sqaod
from sqaod.common.bipartite_graph_bf_searcher_base import BipartiteGraphBFSearcherBase
from sqaod.common import docstring
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
    """ factory function for sqaod.cuda.BipartiteGraphAnnealer.

    Args:
      numpy.ndarray b0, b1, W : QUBO
      optimize : specify optimize direction, `sqaod.maximize or sqaod.minimize <preference.html#sqaod-maximize-sqaod-minimize>`_.
      prefs : `preference <preference.html>`_ as \*\*kwargs
    Returns:
      sqaod.cuda.BipartiteGraphBFSearcher: annealer instance
    """
    searcher = BipartiteGraphBFSearcher(b0, b1, W, optimize, dtype, prefs)
    return searcher

# inherit docstring from interface
docstring.inherit(BipartiteGraphBFSearcher, sqaod.py.BipartiteGraphBFSearcher)

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
