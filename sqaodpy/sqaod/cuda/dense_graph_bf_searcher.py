from __future__ import print_function
import numpy as np
import sqaod
from sqaod.common.dense_graph_bf_searcher_base import DenseGraphBFSearcherBase
from . import cuda_dg_bf_searcher as cext
from . import device

class DenseGraphBFSearcher(DenseGraphBFSearcherBase) :

    def __init__(self, W, optimize, dtype, prefdict) :
        self._cobj = cext.new(dtype)
        cext.assign_device(self._cobj, device.active_device._cobj, dtype)
        self._device = device.active_device
        DenseGraphBFSearcherBase.__init__(self, cext, dtype, W, optimize, prefdict)


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

    N = 20
    W = sqaod.generate_random_symmetric_W(N, -0.5, 0.5, dtype);
    bf = dense_graph_bf_searcher(W, sqaod.minimize, dtype)
    # bf.set_preferences(tile_size = 256)
    
    import time

    start = time.time()
    bf.search()
    end = time.time()
    print(end - start)
    
    x = bf.get_x() 
    E = bf.get_E()
    print(E)
    print(x)

#    import sqaod.py.formulas
#    E = np.empty((1), dtype)
#    for bits in x :
#        print sqaod.py.formulas.dense_graph_calculate_E(W, bits)
