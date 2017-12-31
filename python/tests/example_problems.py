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

def dense_graph_random(N, dtype) :
    return sqaod.generate_random_symmetric_W(N, dtype=dtype)

def bipartite_graph_random(N0, N1, dtype = np.float64) :
    h0 = sqaod.clone_as_np_buffer(np.random.random((N0)) - 0.5, dtype)
    h1 = sqaod.clone_as_np_buffer(np.random.random((N1)) - 0.5, dtype)
    W = sqaod.clone_as_np_buffer(np.random.random((N1, N0)) - 0.5, dtype)
    return h0, h1, W
