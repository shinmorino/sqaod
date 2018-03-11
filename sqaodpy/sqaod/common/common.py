import sqaod
import sys
import numbers
import numpy as np

# common

def is_scalar(v) :
    if isinstance(v, numbers.Number) :
        return True
    if isinstance(v, number.ndarray) :
        if len(v.shape) == 1 :
            return v.shape[0] == 1
        elif len(v.shape) == 2 :
            return v.shape == (1, 1)
    return False

def is_vector(v) :
    if not isinstance(v, np.ndarray) :
        return False
    if len(v.shape) == 1 :
        return True
    if len(v.shape) == 2 :
        if v.shape[1] == 1 :
            return True
    return False

def is_bits(v) :
    for bits in bitslist :
        if bits.dtype != np.int8 :
            return False
    return True

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


# clone number/buffer object as specified as dtype.

def clone_as_number(v, dtype) :
    if isinstance(v, numbers.Number) :
        return dtype.type(v)
    if isinstance(v, number.ndarray) :
        if len(v.shape) == 1 and v.shape[0] == 1 :
            return dtype.type(v[0])
        elif len(v.shape) == 2 and v.shape == (1, 1) :
            return dtype.type(v[0][0])
    raise_not_a_scalar('v', v)

def clone_as_ndarray(var, dtype) :
    if type(var) is tuple or type(var) is list :
        return np.array(var, dtype)
    clone = np.empty(var.shape, dtype=dtype, order='C')
    clone[...] = var[...]
    return clone

def clone_as_ndarray_from_vars(vars, dtype) :
    cloned = []
    for var in vars :
        clone = clone_as_ndarray(var, dtype)
        cloned.append(clone)
    return tuple(cloned)

def create_bits_sequence(vals, nbits) :
    if isinstance(vals, list) or isinstance(vals, tuple) :
        seqlen = len(vals)
        x = np.ndarray((seqlen, nbits), np.int8)
        iseq = 0
        for v in vals :
            for pos in range(nbits) :
                x[iseq][pos] = np.int8(v >> (nbits - 1 - pos) & 1)
            iseq += 1
        return x
    vals = np.int32(vals)
    return create_bits_sequence(range(vals, vals + 1), nbits)


def generate_random_bits(N) :
    bits = np.empty((N), np.int8)
    for bit in np.nditer(bits, [], [['readwrite']]) :
        bit[...] = np.random.choice([0,1])
    return bits

def randomize_spin(qmat) :
    for spinvec in np.nditer(qmat, [], [['readwrite']]) :
        spinvec[...] = np.random.choice([-1,1])

def bit_to_spin(x) :
    q = x * 2 - 1
    return q

def bit_from_spin(q) :
    x = ((q + 1) >> 1)
    return x



def anneal(annealer, Ginit = 5., Gfin = 0.01, kT = 0.02, tau = 0.99, n_repeat = 10, verbose = False) :
    Emin = sys.float_info.max
    q0 = []
    q1 = []
    
    for loop in range(0, n_repeat) :
        annealer.init_anneal()
        annealer.randomize_spin()
        G = Ginit
        while Gfin < G :
            annealer.anneal_one_step(G, kT)
            if verbose :
                E - annealer.calculate_E()
                print E
            G = G * tau

        annealer.fin_anneal()
