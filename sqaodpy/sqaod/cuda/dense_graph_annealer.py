from __future__ import print_function
import numpy as np
import sqaod
from sqaod.common.dense_graph_annealer_base import DenseGraphAnnealerBase
from . import cuda_dg_annealer as cext
from . import device

class DenseGraphAnnealer(DenseGraphAnnealerBase) :

    def __init__(self, W, optimize, dtype, prefdict) :
        self._cobj = cext.new(dtype)
        cext.assign_device(self._cobj, device.active_device._cobj, dtype)
        self._device = device.active_device
        DenseGraphAnnealerBase.__init__(self, cext, dtype, W, optimize, prefdict)

def dense_graph_annealer(W = None, optimize=sqaod.minimize, dtype=np.float64, **prefs) :
    ann = DenseGraphAnnealer(W, optimize, dtype, prefs)
    return ann


if __name__ == '__main__' :

    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]])
    np.random.seed(0)

    N = 8
    m = 4

    N = 100
    m = 5
    W = sqaod.generate_random_symmetric_W(N, -0.5, 0.5, np.float64)

    ann = dense_graph_annealer(W, dtype=np.float64, n_trotters = m)
    import sqaod.py as py
    #ann = py.dense_graph_annealer(W, n_trotters = m)
    ann.set_qubo(W, sqaod.minimize)
    h, J, c = ann.get_hamiltonian()
    print(h)
    print(J)
    print(c)

    
    Ginit = 5.
    Gfin = 0.001
    
    nRepeat = 2
    beta = 1. / 0.02
    tau = 0.995
    
    for loop in range(0, nRepeat) :
        G = Ginit
        ann.prepare()
        ann.randomize_spin()
        while Gfin < G :
            ann.anneal_one_step(G, beta)
            G = G * tau
        ann.make_solution()
        E = ann.get_E()
        q = ann.get_q() 
        print(E)
        print(q)
