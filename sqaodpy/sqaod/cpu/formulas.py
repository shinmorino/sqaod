from __future__ import print_function
from sqaod.common import formulas_base as base
from . import cpu_formulas as cext

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


if __name__ == '__main__' :
    import sqaod
    import sqaod.py.formulas as py_formulas
    import numpy as np
    dtype = np.float64

    np.random.seed(0)

    try :
        W = np.ones((4, 4), np.int32)
        dense_graph_calculate_hamiltonian(W, np.int32)
    except Exception as e :
        print(e.message)

    N0 = 4
    N1 = 3
    W = np.random.random((N1, N0))
    b0 = np.random.random((N0))
    b1 = np.random.random((N1))
    
    x0 = sqaod.generate_random_bits(N0)
    x1 = sqaod.generate_random_bits(N1)

    E0 = py_formulas.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1)
    E1 = bipartite_graph_calculate_E(b0, b1, W, x0, x1, dtype)
    assert np.allclose(E0, E1), "{0} (1)".format((str(E0), str(E1)))
    
    xlist0 = sqaod.create_bitset_sequence(range(0, 1 << N0), N0)
    xlist1 = sqaod.create_bitset_sequence(range(0, 1 << N1), N1)
    E0 = py_formulas.bipartite_graph_batch_calculate_E_2d(b0, b1, W, xlist0, xlist1)
    E1 = bipartite_graph_batch_calculate_E_2d(b0, b1, W, xlist0, xlist1, dtype)
    assert np.allclose(E0, E1)
