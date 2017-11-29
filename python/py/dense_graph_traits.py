import numpy as np

def is_symmetric(mat) :
    return np.allclose(mat, mat.T)

def generate_random_W(N, wmin = -0.5, wmax = 0.5, dtype=np.float64) :
    W = np.zeros((N, N), dtype)
    for i in range(0, N) :
        for j in range(i, N) :
            W[i, j] = np.random.random()
    W = W + np.tril(np.ones((N, N)), -1) * W.T
    W = W * (wmax - wmin) + wmin
    return W

def calculate_hJc(W) :
    if (not is_symmetric(W)) :
        raise Exception('W is not symmetric.')

    N = W.shape[0]
    h = np.ndarray((N), dtype=np.float64)
    W4 = np.ndarray((N, N), dtype=np.float64)
    W4 = 0.25 * W

    for i in range(N) :
        h[i] = 0.5 * np.sum(W[i])
    c = np.sum(W4) + np.sum(W4.diagonal())

    J = W4
    for i in range(0, N) :
        J[i][i] = 0

    return h, J, c

def calculate_qubo_E(W, x) :
    # FIXME: allow x as an array, not matrix.
    return np.sum(x.T * np.matmul(W, x.T), 0)

def calculate_E_from_hJc(h, J, c, q) :
    # FIXME: allow x as  array, not matrix.
    return c + np.matmul(h, q.T) + np.sum(q.T * np.matmul(J, q.T), 0)
