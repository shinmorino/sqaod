import sys
import numbers
import numpy as np

import sys
this = sys.modules[__name__]

this.rtol_fp64 = 1.0e-5
this.atol_fp64 = 1.0e-8
this.rtol_fp32 = 1.0e-4
this.atol_fp32 = 1.0e-5

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

def is_symmetric(mat, rtol = this.rtol_fp64, atol = this.atol_fp64) :
    this = sys.modules[__name__]
    if mat.dtype == np.float64 :
        rtol, atol = this.rtol_fp64, this.atol_fp64
    else :
        rtol, atol = this.rtol_fp32, this.atol_fp32
    return np.allclose(mat, mat.T, rtol, atol)

def is_triangular(mat) :
    atol = this.rtol_fp64 if mat.dtype == np.float64 else this.rtol_fp32
    if not np.any(np.triu(mat, 1) > atol) :
        return True
    return not np.any(np.tril(mat, -1) > atol)

def symmetrize(mat) :
    if is_symmetric(mat) :
        return mat
    
    if is_triangular(mat) :
        d = np.diag(mat)
        mat = mat + mat.T
        N = mat.shape[0]
        mat[range(N), range(N)] -= d
        return mat
    
    raise RuntimeError("given matrix is not triangular nor symmetric.")
    

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

def create_bitset_sequence(vals, nbits) :
    seqlen = len(vals)
    x = np.ndarray((seqlen, nbits), np.int8)
    iseq = 0
    for v in vals :
        for pos in range(nbits) :
            x[iseq][pos] = np.int8(v >> (nbits - 1 - pos) & 1)
        iseq += 1
    return x

def _fix_ndarray_type(obj, dtype) :
    if obj.dtype == dtype and obj.flags['C_CONTIGUOUS'] :
        return obj
    return np.asarray(obj, dtype=dtype, order='C')

def fix_type(obj, dtype) :
    if isinstance(obj, np.ndarray) :
        return _fix_ndarray_type(obj, dtype)
    try :
        objs = []
        for nobj in obj :
            if isinstance(nobj, np.ndarray) :
                nobj = _fix_ndarray_type(nobj, dtype)
            else :
                # try creating ndarray from given object.
                nobj = np.asarray(nobj, dtype=dtype, order='C')
            objs.append(nobj)
        return objs
    except TypeError as te :
        raise RuntimeError('Fix failed.')
    

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



def anneal(annealer, Ginit = 5., Gfin = 0.01, beta = 1. / 0.02, tau = 0.99, n_repeat = 10, verbose = False) :
    Emin = sys.float_info.max
    q0 = []
    q1 = []
    
    for loop in range(0, n_repeat) :
        annealer.prepare()
        annealer.randomize_spin()
        G = Ginit
        while Gfin < G :
            annealer.anneal_one_step(G, beta)
            if verbose :
                E - annealer.calculate_E()
                print(E)
            G = G * tau

        annealer.make_solution()
