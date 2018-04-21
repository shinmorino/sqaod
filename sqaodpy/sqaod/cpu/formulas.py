from __future__ import print_function
import numpy as np
import numbers
import sqaod
from sqaod.common import checkers
import sqaod.common as common
from . import cpu_formulas

# dense graph    

# QUBO energy functions

def dense_graph_calculate_E(W, x, dtype) :
    W = common.fix_precision(W, dtype)
    checkers.dense_graph.qubo(W, dtype)
    checkers.dense_graph.bits(W, x)
    checkers.assert_is_vector('x', x)

    E = np.ndarray((1), dtype)
    cpu_formulas.dense_graph_calculate_E(E, W, x, dtype)
    return E[0]

def dense_graph_batch_calculate_E(W, x, dtype) :
    W = common.fix_precision(W, dtype)
    checkers.dense_graph.qubo(W, dtype);
    checkers.dense_graph.bits(W, x)

    E = np.empty((x.shape[0]), dtype)
    cpu_formulas.dense_graph_batch_calculate_E(E, W, x, dtype)
    return E


# QUBO -> Ising model

def dense_graph_calculate_hamiltonian(W, dtype) :
    W = common.fix_precision(W, dtype)
    checkers.dense_graph.qubo(W, dtype)

    N = W.shape[0]
    h = np.empty((N), dtype)
    J = np.empty((N, N), dtype)
    c = np.empty((1), dtype)
    cpu_formulas.dense_graph_calculate_hamiltonian(h, J, c, W, dtype);
    return h, J, c[0]

# Ising model energy functions

def dense_graph_calculate_E_from_spin(h, J, c, q, dtype) :
    h, J = common.fix_precision([h, J], dtype)
    checkers.dense_graph.hJc(h, J, c, dtype);
    checkers.dense_graph.bits(J, q);
    checkers.assert_is_vector('q', q)
    
    E = np.ndarray((1), dtype)
    cpu_formulas.dense_graph_calculate_E_from_spin(E, h, J, c, q, dtype)
    return E[0]

def dense_graph_batch_calculate_E_from_spin(h, J, c, q, dtype) :
    h, J = common.fix_precision([h, J], dtype)
    checkers.dense_graph.hJc(h, J, c, dtype);
    checkers.dense_graph.bits(J, q);

    E = np.empty([q.shape[0]], dtype)
    cpu_formulas.dense_graph_batch_calculate_E_from_spin(E, h, J, c, q, dtype)
    return E


# bipartite_graph

def bipartite_graph_calculate_E(b0, b1, W, x0, x1, dtype) :
    b0, b1, W = common.fix_precision([b0, b1, W], dtype)
    checkers.bipartite_graph.qubo(b0, b1, W, dtype)
    checkers.bipartite_graph.bits(W, x0, x1)
    checkers.assert_is_vector('x0', x0)
    checkers.assert_is_vector('x1', x1)
    E = np.ndarray((1), dtype)
    cpu_formulas.bipartite_graph_calculate_E(E, b0, b1, W, x0, x1, dtype)
    return E[0]


def bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, dtype) :
    b0, b1, W = common.fix_precision([b0, b1, W], dtype)
    checkers.bipartite_graph.qubo(b0, b1, W, dtype)
    checkers.bipartite_graph.bits(W, x0, x1)
    # FIXME: fix error messages.  move to checkers.py?
    nBatch0 = 1 if len(x0.shape) == 1 else x0.shape[0]
    nBatch1 = 1 if len(x1.shape) == 1 else x1.shape[0]
    if nBatch0 != nBatch1 :
        raise Exception("Different batch dims between x0 and x1.")
    
    nBatch = 1 if len(x0.shape) == 1 else x0.shape[0]
    E = np.empty((nBatch), dtype)
    cpu_formulas.bipartite_graph_batch_calculate_E(E, b0, b1, W, x0, x1, dtype)
    return E

def bipartite_graph_batch_calculate_E_2d(b0, b1, W, x0, x1, dtype) :
    b0, b1, W = common.fix_precision([b0, b1, W], dtype)
    checkers.bipartite_graph.qubo(b0, b1, W, dtype)
    checkers.bipartite_graph.bits(W, x0, x1)
    
    nBatch0 = 1 if len(x0.shape) == 1 else x0.shape[0]
    nBatch1 = 1 if len(x1.shape) == 1 else x1.shape[0]
    E = np.empty((nBatch1, nBatch0), dtype)
    cpu_formulas.bipartite_graph_batch_calculate_E_2d(E, b0, b1, W, x0, x1, dtype)
    return E


def bipartite_graph_calculate_hamiltonian(b0, b1, W, dtype) :
    b0, b1, W = common.fix_precision([b0, b1, W], dtype)
    checkers.bipartite_graph.qubo(b0, b1, W, dtype)

    N0 = W.shape[1]
    N1 = W.shape[0]
    h0 = np.empty((N0), dtype)
    h1 = np.empty((N1), dtype)
    J = np.empty((N1, N0), dtype)
    c = np.empty((1), dtype)
    cpu_formulas.bipartite_graph_calculate_hamiltonian(h0, h1, J, c, b0, b1, W, dtype);
    return h0, h1, J, c[0]

def bipartite_graph_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype) :
    h0, h1, W = common.fix_precision([h0, h1, J], dtype)
    checkers.bipartite_graph.hJc(h0, h1, J, c, dtype)
    checkers.bipartite_graph.bits(J, q0, q1)
    checkers.assert_is_vector('q0', q0)
    checkers.assert_is_vector('q1', q1)

    E = np.ndarray((1), dtype)
    cpu_formulas.bipartite_graph_calculate_E_from_spin(E, h0, h1, J, c, q0, q1, dtype)
    return E[0]


def bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype) :
    h0, h1, J = common.fix_precision([h0, h1, J], dtype)
    checkers.bipartite_graph.hJc(h0, h1, J, c, dtype)
    checkers.bipartite_graph.bits(J, q0, q1)
    
    nBatch0 = 1 if len(q0.shape) == 1 else q0.shape[0]
    nBatch1 = 1 if len(q1.shape) == 1 else q1.shape[0]
    E = np.empty((nBatch0), dtype)
    cpu_formulas.bipartite_graph_batch_calculate_E_from_spin(E, h0, h1, J, c, q0, q1, dtype)
    return E


if __name__ == '__main__' :
    import sqaod
    import sqaod.py.formulas as py_formulas
    dtype = np.float64

    np.random.seed(0)

    try :
        W = np.ones((4, 4), np.int32)
        dense_graph_calculate_hamiltonian(W, np.int32)
    except Exception as e :
        print(str(e))
    
    # dense graph
    N = 16
    W = sqaod.generate_random_symmetric_W(N)

    x = sqaod.generate_random_bits(N)
    E0 = py_formulas.dense_graph_calculate_E(W, x)
    E1 = dense_graph_calculate_E(W, x, dtype)
    assert np.allclose(E0, E1)

    xlist = sqaod.create_bitset_sequence(range(0, 1 << N), N)
    E0 = py_formulas.dense_graph_batch_calculate_E(W, xlist)
    E1 = dense_graph_batch_calculate_E(W, xlist, dtype)
    assert np.allclose(E0, E1)

    h0, J0, c0 = py_formulas.dense_graph_calculate_hamiltonian(W)
    h1, J1, c1 = dense_graph_calculate_hamiltonian(W, dtype)
    assert np.allclose(h0, h1);
    assert np.allclose(J0, J1);
    assert np.allclose(c0, c1);

    q = sqaod.bit_to_spin(x)
    E0 = py_formulas.dense_graph_calculate_E_from_spin(h0, J0, c0, q);
    E1 = dense_graph_calculate_E_from_spin(h0, J0, c0, q, dtype);
    assert np.allclose(E0, E1)

    qlist = sqaod.bit_to_spin(xlist)
    E0 = py_formulas.dense_graph_batch_calculate_E_from_spin(h0, J0, c0, qlist);
    E1 = dense_graph_batch_calculate_E_from_spin(h0, J0, c0, qlist, dtype);
    assert np.allclose(E0, E1)

    
    # rbm

    N0 = 4
    N1 = 3
    W = np.random.random((N1, N0))
    b0 = np.random.random((N0))
    b1 = np.random.random((N1))
    
    x0 = sqaod.generate_random_bits(N0)
    x1 = sqaod.generate_random_bits(N1)

    E0 = py_formulas.bipartite_graph_calculate_E(b0, b1, W, x0, x1)
    E1 = bipartite_graph_calculate_E(b0, b1, W, x0, x1, dtype)
    assert np.allclose(E0, E1), "{0} (1)".format((str(E0), str(E1)))

    E0 = py_formulas.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1)
    E1 = bipartite_graph_calculate_E(b0, b1, W, x0, x1, dtype)
    assert np.allclose(E0, E1), "{0} (1)".format((str(E0), str(E1)))
    
    xlist0 = sqaod.create_bitset_sequence(range(0, 1 << N0), N0)
    xlist1 = sqaod.create_bitset_sequence(range(0, 1 << N1), N1)
    E0 = py_formulas.bipartite_graph_batch_calculate_E_2d(b0, b1, W, xlist0, xlist1)
    E1 = bipartite_graph_batch_calculate_E_2d(b0, b1, W, xlist0, xlist1, dtype)
    assert np.allclose(E0, E1)
  
    h00, h01, J0, c0 = py_formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W)
    h10, h11, J1, c1 = bipartite_graph_calculate_hamiltonian(b0, b1, W, dtype)
    assert np.allclose(h00, h10);
    assert np.allclose(h01, h11);
    assert np.allclose(J0, J1);
    assert np.allclose(c0, c1);

    q0 = sqaod.bit_to_spin(x0)
    q1 = sqaod.bit_to_spin(x1)
    E0 = py_formulas.bipartite_graph_calculate_E_from_spin(h00, h01, J0, c0, q0, q1);
    E1 = bipartite_graph_calculate_E_from_spin(h10, h11, J0, c0, q0, q1, dtype);
    assert np.allclose(E0, E1)
    
    xlist0 = sqaod.create_bitset_sequence(range(0, 1 << N0), N0)
    xlist1 = sqaod.create_bitset_sequence(range(0, 1 << N0), N1)
    qlist0 = sqaod.bit_to_spin(xlist0)
    qlist1 = sqaod.bit_to_spin(xlist1)
    E0 = py_formulas.bipartite_graph_batch_calculate_E_from_spin(h10, h11, J0, c0, qlist0, qlist1);
    E1 = bipartite_graph_batch_calculate_E_from_spin(h10, h11, J0, c0, qlist0, qlist1, dtype);
    assert np.allclose(E0, E1), "{0} (1)".format((str(E0), str(E1)))
    

    #W = np.ones((3, 3))
    #h0, J0, c0 = py_formulas.dense_graph_calculate_hamiltonian(W)
    #h1, J1, c1 = dense_graph_calculate_hamiltonian(W, dtype)
        

    """
    print(q)
    print(E0)
    print(E1)
    """    
