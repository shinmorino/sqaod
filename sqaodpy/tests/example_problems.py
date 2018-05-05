import numpy as np
import sqaod

def dense_graph_8x8(dtype) :
    
    W = np.array([[-32,4,4,4,4,4,4,4],
                  [4,-32,4,4,4,4,4,4],
                  [4,4,-32,4,4,4,4,4],
                  [4,4,4,-32,4,4,4,4],
                  [4,4,4,4,-32,4,4,4],
                  [4,4,4,4,4,-32,4,4],
                  [4,4,4,4,4,4,-32,4],
                  [4,4,4,4,4,4,4,-32]], dtype)
    return W

def _quantize(W, dtype) :
    Q = np.rint(W * 16384) / 16384.
    return np.asarray(Q, dtype)

def dense_graph_random(N, dtype) :
    W = sqaod.generate_random_symmetric_W(N, dtype=dtype)
    return _quantize(W, dtype)


def bipartite_graph_random(N0, N1, dtype = np.float64) :
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    W = np.random.random((N1, N0)) - 0.5
    return _quantize(b0, dtype), _quantize(b1, dtype), _quantize(W, dtype)
