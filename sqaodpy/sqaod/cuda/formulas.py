from __future__ import print_function
import sys
from sqaod.common import formulas_base as base
from . import device
from . import cuda_formulas as cext

# dense graph

def dense_graph_calculate_E(W, x, dtype) :
    return base.dense_graph_calculate_E(cext, _dgobj, W, x, dtype)

def dense_graph_batch_calculate_E(W, x, dtype) :
    return base.dense_graph_batch_calculate_E(cext, _dgobj, W, x, dtype)

def dense_graph_calculate_hamiltonian(W, dtype) :
    return base.dense_graph_calculate_hamiltonian(cext, _dgobj, W, dtype)

def dense_graph_calculate_E_from_spin(h, J, c, q, dtype) :
    return base.dense_graph_calculate_E_from_spin(cext, _dgobj, h, J, c, q, dtype)

def dense_graph_batch_calculate_E_from_spin(h, J, c, q, dtype) :
    return base.dense_graph_batch_calculate_E_from_spin(cext, _dgobj, h, J, c, q, dtype)


# bipartite_graph

def bipartite_graph_calculate_E(b0, b1, W, x0, x1, dtype) :
    return base.bipartite_graph_calculate_E(cext, _bgobj, b0, b1, W, x0, x1, dtype)

def bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, dtype) :
    return base.bipartite_graph_batch_calculate_E(cext, _bgobj, b0, b1, W, x0, x1, dtype)

def bipartite_graph_batch_calculate_E_2d(b0, b1, W, x0, x1, dtype) :
    return base.bipartite_graph_batch_calculate_E_2d(cext, _bgobj, b0, b1, W, x0, x1, dtype)

def bipartite_graph_calculate_hamiltonian(b0, b1, W, dtype) :
    return base.bipartite_graph_calculate_hamiltonian(cext, _bgobj, b0, b1, W, dtype)

def bipartite_graph_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype) :
    return base.bipartite_graph_calculate_E_from_spin(cext, _bgobj, h0, h1, J, c, q0, q1, dtype)

def bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype) :
    return base.bipartite_graph_batch_calculate_E_from_spin(cext, _bgobj, h0, h1, J, c, q0, q1, dtype)


# Global/Static

this_module = sys.modules[__name__]

def unload() :
    cext.dg_formulas_delete(this_module._dgobj)
    cext.bg_formulas_delete(this_module._bgobj)
    this_module._dgobj = None
    this_module._bgobj = None

if __name__ != "__main__" :
    this_module._dgobj = cext.dg_formulas_new()
    this_module._bgobj = cext.bg_formulas_new()
    # initialization
    cext.dg_formulas_assign_device(this_module._dgobj, device.active_device._cobj)
    cext.bg_formulas_assign_device(this_module._bgobj, device.active_device._cobj)
    import atexit
    atexit.register(unload)
