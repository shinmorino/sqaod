import numpy as np
import sqaod
from sqaod.common import checkers, symmetrize


def dense_graph_calculate_hamiltonian(W, dtype = None) :
    """ Calculate hamiltonian from given QUBO.

    Args:
      numpy.ndarray W : QUBO, W should be a upper/lower triangular or symmetric matrix.
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      tuple: tuple containing Hamiltonian.
        h(vector as numpy.array), J(2-D symmetric matrix as numpy.array), c(scalar value)
    """
 
    checkers.dense_graph.qubo(W)
    W = symmetrize(W)

    N = W.shape[0]
    h = np.ndarray((N), dtype=np.float64)
    W4 = - 0.25 * W

    for i in range(N) :
        h[i] = - 0.5 * np.sum(W[i])
    c = np.sum(W4) + np.sum(W4.diagonal())

    J = W4
    for i in range(0, N) :
        J[i][i] = 0

    return h, J, c


def dense_graph_calculate_E(W, x, dtype = None) :
    """ Calculate desne graph QUBO energy from bits.

    Args:
      numpy.array W: QUBO, W should be a upper/lower triangular or symmetric matrix.
      numpy.ndarray x : array of bit {0, 1}.
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      QUBO energy
    """
    
    checkers.dense_graph.qubo(W)
    checkers.dense_graph.x(W, x)
    W = symmetrize(W)

    if len(x.shape) != 1 :
        if x.shape[0] != 1 :
            raise Exception('Wrong dimention of x')
        x = x.reshape(-1)
    return np.dot(x, np.matmul(W, x.T))

def dense_graph_batch_calculate_E(W, x, dtype = None) :
    """ Batched version of the function to calculate dense graph QUBO energy from bits.

    Args:
      numpy.ndarray W: QUBO, W should be a upper/lower triangular or symmetric matrix.
      numpy.ndarray x:
        bit arrays repsesented as 2-D matrix(n_trotters x n_bits).
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64

    Returns:
      QUBO energy
    """
    
    checkers.dense_graph.qubo(W)
    checkers.dense_graph.xbatch(W, x)
    W = symmetrize(W)
    
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    return np.sum(x * np.matmul(W, x.T).T, 1)

def dense_graph_calculate_E_from_spin(h, J, c, q, dtype = None) :
    """ Calculate desne graph QUBO energy from spins.

    Args:
      (numpy.ndarray) h, J, c : Hamiltonian h(1-D vector), W(sqauare matrix), c(scalar).
        W must be a upper/lower triangular or symmetric matrix.
      q (numpy.ndarray) : array of spin {-1, 1}.
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64

    Returns:
      floating point number : QUBO energy
    """
    checkers.dense_graph.hJc(h, J, c)
    checkers.dense_graph.q(J, q)
    J = symmetrize(J)
    
    return - c - np.dot(h, q) - np.dot(q, np.matmul(J, q.T))

def dense_graph_batch_calculate_E_from_spin(h, J, c, q, dtype = None) :
    """ Batched version of the function to calculate dense graph QUBO energy from spins.

    Args:
      numpy.ndarray h, J, c : Hamiltonian h(1-D vector), W(sqauare matrix), c(scalar).
        W must be a upper/lower triangular or symmetric matrix.
      numpy.ndarray q :
        bit arrays repsesented as 2-D matrix(n_trotters x n_bits).
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      QUBO energy
    """
    checkers.dense_graph.hJc(h, J, c)
    checkers.dense_graph.qbatch(J, q)
    J = symmetrize(J)
    
    if len(q.shape) == 1:
        q = q.reshape(1, -1)
    return - c - np.matmul(h, q.T) - np.sum(q.T * np.matmul(J, q.T), 0)


# bibartite graph

def bipartite_graph_calculate_hamiltonian(b0, b1, W, dtype = None) :
    """ Calculate hamiltonian from given QUBO.

    Args:
      numpy.ndarray b0, b1, W : Bipartite graph QUBO.
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64

    Returns:
      tuple: tuple containing Hamiltonian.
        h0(1-D numpy.array), h1(1-D numpy.array), J(matrix as numpy.array), c(scalr value).
    """
    checkers.bipartite_graph.qubo(b0, b1, W);
    N0 = W.shape[1]
    N1 = W.shape[0]
    
    c = - 0.25 * np.sum(W) - 0.5 * (np.sum(b0) + np.sum(b1))
    J = - 0.25 * W
    h0 = np.empty((N0), W.dtype)
    h1 = np.empty((N1), W.dtype)
    for i in range(N0) :
        h0[i] = (- 1. / 4.) * np.sum(W[:, i]) - 0.5 * b0[i]
    for j in range(N1) :
        h1[j] = (- 1. / 4.) * np.sum(W[j]) - 0.5 * b1[j]

    return h0, h1, J, c

def bipartite_graph_calculate_E(b0, b1, W, x0, x1, dtype = None) :
    """ Calculate bipartite graph QUBO energy from bits.

    Args:
      numpy.array b0, b1, W: QUBO.  b0(vector as numpy.array), b1(vector as numpy.array), W(matrix as numpy.array).  W should be a upper/lower triangular or symmetric matrix.
      numpy.ndarray x0, x1 : array of bit {0, 1}.
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      QUBO energy
    """
    checkers.bipartite_graph.qubo(b0, b1, W)
    checkers.bipartite_graph.x(W, x0, x1)
    return np.dot(b0, x0) + np.dot(b1, x1) + np.dot(x1, np.matmul(W, x0))

def bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, dtype = None) :
    """ Batched version of the function to calculate dense graph QUBO energy from bits.

    Args:
      numpy.array b0, b1, W: QUBO.  b0(vector as numpy.array), b1(vector as numpy.array), W(matrix as numpy.array).  W should be a upper/lower triangular or symmetric matrix.
      numpy.ndarray x0, x1:
        bit arrays repsesented as 2-D matrix.
        x0 shape is n_trotters x n_bits_0, and x1 shape is n_trotters x n_bits_1
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      QUBO energy
    """
    if len(x0.shape) == 1 :
        x0 = x0.reshape(1, -1)
    if len(x1.shape) == 1 :
        x1 = x1.reshape(1, -1)
    checkers.bipartite_graph.qubo(b0, b1, W)
    checkers.bipartite_graph.xbatch(W, x0, x1)
    
    nBatch0, nBatch1 = x0.shape[0], x1.shape[0]
    if nBatch0 != nBatch1 :
        raise Exception("Different batch dims between x0 and x1.")
    return np.matmul(b0, x0.T) + np.matmul(b1, x1.T) + np.sum(x1 * np.matmul(W, x0.T).T, 1)

def bipartite_graph_batch_calculate_E_2d(b0, b1, W, x0, x1, dtype = None) :
    """ 2D batched version of the function to calculate dense graph QUBO energy from bits.

    Args:
      numpy.array b0, b1, W: QUBO.  b0(vector as numpy.array), b1(vector as numpy.array), W(matrix as numpy.array).  W should be a upper/lower triangular or symmetric matrix.
      numpy.ndarray x0, x1:
        bit arrays repsesented as 2-D matrix.
        x0 shape is n_trotters x n_bits_0, and x1 shape is n_trotters x n_bits_1
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      QUBO energy as a matrix, whose shape is (len(x1), (len(x0)).

    This function calculates QUBO energy for all combinations of elements x0 and x1, 
    used in bipartite graph brute-force searchers.

    """
    if len(x0.shape) == 1 :
        x0 = x0.reshape(1, -1)
    if len(x1.shape) == 1 :
        x1 = x1.reshape(1, -1)
    checkers.bipartite_graph.qubo(b0, b1, W)
    checkers.bipartite_graph.xbatch(W, x0, x1)
    
    return np.matmul(b0.T, x0.T).reshape(1, -1) + np.matmul(b1.T, x1.T).reshape(-1, 1) \
        + np.matmul(x1, np.matmul(W, x0.T))

def bipartite_graph_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype = None) :
    """ Calculate bipartite graph QUBO energy from spins.

    Args:
      numpy.array b0, b1, W: QUBO.  b0(vector as numpy.array), b1(vector as numpy.array), W(matrix as numpy.array).  W should be a upper/lower triangular or symmetric matrix.
      numpy.ndarray x0, x1 : array of bit {0, 1}.
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      QUBO energy
    """
    checkers.bipartite_graph.hJc(h0, h1, J, c)
    checkers.bipartite_graph.q(J, q0, q1)
    
    return - np.dot(h0, q0) - np.dot(h1, q1) - np.dot(q1, np.matmul(J, q0)) - c

def bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c, q0, q1, dtype = None) :
    """ Batched version of the function to calculate dense graph QUBO energy from bits.

    Args:
      numpy.array b0, b1, W: QUBO. b0(vector as numpy.array), b1(vector as numpy.array), W(matrix as numpy.array).  W should be a upper/lower triangular or symmetric matrix.
      numpy.ndarray x0, x1:
        bit arrays repsesented as 2-D matrix.
        x0 shape is n_trotters x n_bits_0, and x1 shape is n_trotters x n_bits_1
      numpy.dtype dtype : Numerical precision, numpy.float32 or numpy.float64.

    Returns:
      QUBO energy
    """
    if len(q0.shape) == 1 :
        q0 = q0.reshape(1, -1)
    if len(q1.shape) == 1 :
        q1 = q1.reshape(1, -1)
    checkers.bipartite_graph.hJc(h0, h1, J, c)
    checkers.bipartite_graph.qbatch(J, q0, q1)
    
    nBatch0, nBatch1 = q0.shape[0], q1.shape[0]
    if nBatch0 != nBatch1 :
        raise Exception("Different batch dims between x0 and x1.")
    return - np.matmul(h0, q0.T) - np.matmul(h1, q1.T) - np.sum(q1.T * np.matmul(J, q0.T), 0) - c
