import sys
import numpy as np

# common

def is_symmetric(mat) :
    return np.allclose(mat, mat.T)

def generate_random_symmetric_W(N, wmin = -0.5, wmax = 0.5, dtype=np.float64) :
    W = np.zeros((N, N), dtype)
    for i in range(0, N) :
        for j in range(i, N) :
            W[i, j] = np.random.random()
    W = W + np.tril(np.ones((N, N)), -1) * W.T
    W = W * (wmax - wmin) + wmin
    return W


def create_bits_sequence(vals, nbits) :
    if isinstance(vals, list) or isinstance(vals, tuple) :
        seqlen = len(vals)
        x = np.ndarray((seqlen, nbits), np.int8)
        iseq = 0
        for v in vals :
            for pos in range(nbits) :
                x[iseq][pos] = np.int8(v >> pos & 1)
            iseq += 1
        return x
    vals = np.int32(vals)
    return create_bits_sequence(range(vals, vals + 1), nbits)


def generate_random_bits(N) :
    bits = np.empty((N), np.int8)
    for bit in np.nditer(bits, [], [['readwrite']]) :
        bit[...] = np.random.choice([0,1])
    return bits

def randomize_qbits(qbits) :
    for qbit in np.nditer(qbits, [], [['readwrite']]) :
        qbit[...] = np.random.choice([-1,1])

def bits_to_qbits(x) :
    q = x * 2 - 1
    return q

def bits_from_qbits(q) :
    x = ((q + 1) >> 1)
    return x


def anneal(annealer, Ginit = 5., Gfin = 0.01, kT = 0.02, tau = 0.99, n_repeat = 10, verbose = False) :
    Emin = sys.float_info.max
    q0 = []
    q1 = []
    
    for loop in range(0, n_repeat) :
        annealer.init_anneal()
        annealer.randomize_q()
        G = Ginit
        while Gfin < G :
            annealer.anneal_one_step(G, kT)
            if verbose :
                E - annealer.calculate_E()
                print E
            G = G * tau

        annealer.calculate_E()
