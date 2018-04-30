from __future__ import print_function
from sqaod.common import formulas_base as base
from . import cuda_formulas as cext
from . import device

# initialization
cext.assign_device(device.active_device._cobj)

# dense graph

def dense_graph_calculate_E(W, x, dtype) :
    return base.dense_graph_calculate_E(W, x, cext, dtype)

def dense_graph_batch_calculate_E(W, x, dtype) :
    return base.dense_graph_batch_calculate_E(W, x, cext, dtype)

def dense_graph_calculate_hamiltonian(W, dtype) :
    return base.dense_graph_calculate_hamiltonian(W, cext, dtype)

def dense_graph_calculate_E_from_spin(h, J, c, q, dtype) :
    return base.dense_graph_calculate_E_from_spin(h, J, c, q, cext, dtype)

def dense_graph_batch_calculate_E_from_spin(h, J, c, q, dtype) :
    return base.dense_graph_batch_calculate_E_from_spin(h, J, c, q, cext, dtype)


# bipartite_graph

def bipartite_graph_calculate_E(b0, b1, W, x0, x1, dtype) :
    return base.bipartite_graph_calculate_E(b0, b1, W, x0, x1, cext, dtype)

def bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, dtype) :
    return base.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, cext, dtype)

def bipartite_graph_batch_calculate_E_2d(b0, b1, W, x0, x1, dtype) :
    return base.bipartite_graph_batch_calculate_E_2d(b0, b1, W, x0, x1, cext, dtype)

def bipartite_graph_calculate_hamiltonian(b0, b1, W, dtype) :
    return base.bipartite_graph_calculate_hamiltonian(b0, b1, W, cext, dtype)

def bipartite_graph_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype) :
    return base.bipartite_graph_calculate_E_from_spin(h0, h1, J, c, q0, q1, cext, dtype)

def bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype) :
    return base.bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c, q0, q1, cext, dtype)
