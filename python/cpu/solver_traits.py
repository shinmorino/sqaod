import numpy as np
import numbers
import cpu_native

# checkers for object types

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
    if len(v.shape) == 1 :
        return True
    if len(v.shape) == 2 :
        if v.shape[1] == 1 :
            return True
    return False

def to_prec(dtype) :
    if isinstance(dtype, np.float64) or dtype is np.float64 :
        return 64
    if isinstance(dtype, np.float32) or dtype is np.float32:
        return 32
    raise Exception("Unexpected dtype, {0}.".format(str(dtype)))


# raising exceptions

def raise_not_a_scalar(caption, var) :
    tokens = [var, ' is not a scalar. ', str(type(var))]
    msg = ''.join(tokens)
    raise Exception(msg)

def raise_not_a_vector(caption, var) :
    tokens = [caption, ' is not a vector. ', str(type(var))]
    msg = ''.join(tokens)
    raise Exception(msg)

def raise_wrong_bit_type(catpion, var) :
    tokens = [var, ' is not a bit type(np.int8). ', str(type(var))]
    msg = ''.join(tokens)
    raise Exception(msg)

def raise_wrong_shape(caption, mat) :
    tokens = ['wrong shape dim=(', caption, ') = ']
    tokens.append(str(mat.shape))
    msg = ''.join(tokens)
    raise Exception(msg)

def raise_dimension_mismatch(caption, varlist) :
    tokens = ['dimension mismatch for ', caption, '. ']
    for var in varlist :
        tokens.append(str(var.shape))
        tokens.append(', ')
    tokens.pop()
    msg = ''.join(tokens)
    raise Exception(msg)


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

def clone_as_np_buffer(buf, dtype) :
    if buf is tuple or buf is list :
        return np.array(buf, dtype)
    return np.ndarray(mat.shape, buffer=mat, order='C')

# value/type checks

def _check_is_bits(bitslist) :
    for bits in bitslist :
        if bits.dtype != np.int8 :
            raise Exception("bits are not given as np.int8")

def _check_is_vector(caption, vec) :
    if not is_vector(vec) :
        raise_not_a_vector(caption, vec)

def _check_buffer_precision(dtype, matlist) :
    for mat in matlist :
        if mat.dtype != dtype :
            raise Exception("precision mismatch.")
        
# dense graph    

# qubo dimension check
def _dg_check_qubo_var_dims(W, x) :
    if len(W.shape) != 2 :
        raise_wrong_shape('W', W)
    # dimension checks
    if not W.shape[0] == W.shape[1] == x.shape[len(x.shape) - 1] :
        raise_dims_dont_match("W, x", (W, x))

# ising model dimension check    
def _dg_check_ising_var_dims(h, J, c, q) :
    if not is_vector(h) :
        raise_not_a_vector('h', h)
    if len(J.shape) != 2 or J.shape[0] != J.shape[1] :
        raise_wrong_shape('J', (J))
    if q.dtype != np.int8 :
        raise_wrong_bit_type('q', (q))
    if not h.shape[0] == J.shape[0] == J.shape[1] == q.shape[len(q.shape) - 1] :
        raise_dims_dont_match("h, J, c", (h, J, c))
    # FIXME: check for c :



# QUBO energy functions

def dense_graph_calculate_E(W, x, dtype) :
    _dg_check_qubo_var_dims(W, x)
    _check_buffer_precision(dtype, (W))
    _check_is_vector('x', x)
    _check_is_bits((x))
    E = np.ndarray((1), dtype)
    cpu_native.dense_graph_calculate_E(E, W, x, to_prec(dtype))
    return E[0]

def dense_graph_batch_calculate_E(W, x, dtype) :
    _dg_check_qubo_var_dims(W, x);
    _check_is_bits((x))
    E = np.empty((x.shape[0]), dtype)
    cpu_native.dense_graph_batch_calculate_E(E, W, x, to_prec(dtype))
    return E


# QUBO -> Ising model

def dense_graph_calculate_hJc(W, dtype) :
    if len(W.shape) != 2 or W.shape[0] != W.shape[1] :
        raise_wrong_shape('W', W)
    _check_buffer_precision(dtype, (W))
    N = W.shape[0]
    h = np.empty((N), dtype)
    J = np.empty((N, N), dtype)
    c = np.empty((1), dtype)
    cpu_native.dense_graph_calculate_hJc(h, J, c, W, to_prec(dtype));
    return h, J, c[0]

# Ising model energy functions

def dense_graph_calculate_E_from_qbits(h, J, c, q, dtype) :
    _dg_check_ising_var_dims(h, J, c, q);
    
    q_valid = (len(q.shape) == 1) or (len(q.shape) == 2 and q.shape[1] != 1)
    if not q_valid :
        raise Exception('q should be a vector(array)')
    E = np.ndarray((1), dtype)
    cpu_native.dense_graph_calculate_E_from_qbits(E, h, J, c, q, to_prec(dtype))
    return E[0]

def dense_graph_batch_calculate_E_from_qbits(h, J, c, q, dtype) :
    _dg_check_ising_var_dims(h, J, c, q);
    E = np.empty([q.shape[0]], dtype)
    cpu_native.dense_graph_batch_calculate_E_from_qbits(E, h, J, c, q, to_prec(dtype))
    return E


# rbm



def _rbm_check_qubo_var_dims(b0, b1, W, x) :
    if not is_vector(b0) :
        raise_not_vector('b0', b0)
    if not is_vector(x) :
        raise_not_vector('b1', b1)
    if len(W.shape) != 2 :
        raise_wrong_shape('W', W)
    if not is_vector(x) :
        raise_not_vector('x', x)
    # dimension checks
    if not W.shape[1] == b0.shape[0] == b1.shape[0] == x.shape[0] :
        raise_dims_dont_match('b0, b1, W, x', (b0, b1, W, x))

def _rbm_check_ising_var_dims(h0, h1, J, c, q) :
    if len(h0.shape) != 1 :
        raise_not_a_vector('h0', h0)
    if len(h1.shape) != 1 :
        raise_not_a_vector('h1', h1)
    if len(J.shape) != 2 and  J.shape[0] != J.shape[1] :
        raise_wrong_shape('J', J)
    if q.dtype != np.int8 :
        raise_wrong_bit_type('q', q)
    if not h.shape[0] == J.shape[0] == J.shape[1] == q.shape[0] :
        raise_dims_dont_match('h, J, c', (h, J, c))
    if is_number(c) :
        raise_not_a_number('c', c)


def rbm_calculate_E(b0, b1, W, x0, x1, dtype) :
    return - np.dot(b0, x0) - np.dot(b1, x1) - np.dot(x1, np.matmul(W, x0))

def rbm_batch_calculate_E(b0, b1, W, x0, x1, dtype) :
    return - np.matmul(b0, x0.T).reshape(1, iStep) - np.matmul(b1, x1.T).reshape(jStep, 1) \
        - np.matmul(x1, np.matmul(W, x0.T))

def rbm_calculate_hJc(W) :
    N0 = W.shape[1]
    N1 = W.shape[0]
    
    c = 0.25 * np.sum(W) + 0.5 * (np.sum(b0) + np.sum(b1))
    J = 0.25 * W
    h0 = [(1. / 4.) * np.sum(W[:, i]) + 0.5 * b0[i] for i in range(0, N0)]
    h1 = [(1. / 4.) * np.sum(W[j]) + 0.5 * b1[j] for j in range(0, N1)]
    hlist = [h0, h1]

    return hlist, J, c

def rbm_calculate_E_from_qbits(h0, h1, J, c, q0, q1, dtype) :
    return - np.dot(h0, q0) - np.dot(h1, q1) - np.dot(q1[0], np.matmul(J, q0[0])) - c

def rbm_batch_calculate_E_from_qbits(h, J, c, q0, q1, dtype) :
    return - np.matmul(h0, q0.T).reshape(1, q0.shape[0]) \
        - np.matmul(h1, q1.T).reshape(1, q1.shape[1]) \
        - np.dot(q1[0], np.matmul(J, q0[0])) - c


if __name__ == '__main__' :
    import utils
    import py.solver_traits
    dtype = np.float64

    # dense graph

    N = 16
    W = utils.generate_random_symmetric_W(N)
    
    x = utils.generate_random_bits(N)
    E0 = py.solver_traits.dense_graph_calculate_E(W, x)
    E1 = dense_graph_calculate_E(W, x, dtype)
    assert np.allclose(E0, E1)

    xlist = utils.create_bits_sequence(range(0, 1 << N), N)
    E0 = py.solver_traits.dense_graph_batch_calculate_E(W, xlist)
    E1 = dense_graph_batch_calculate_E(W, xlist, dtype)
    assert np.allclose(E0, E1)

    h0, J0, c0 = py.solver_traits.dense_graph_calculate_hJc(W)
    h1, J1, c1 = dense_graph_calculate_hJc(W, dtype)
    assert np.allclose(h0, h1);
    assert np.allclose(J0, J1);
    assert np.allclose(c0, c1);
    
    q = utils.bits_to_qbits(x)
    E0 = py.solver_traits.dense_graph_calculate_E_from_qbits(h0, J0, c0, q);
    E1 = dense_graph_calculate_E_from_qbits(h0, J0, c0, q, dtype);
    assert np.allclose(E0, E1)
    
    qlist = utils.bits_to_qbits(xlist)
    E0 = py.solver_traits.dense_graph_batch_calculate_E_from_qbits(h0, J0, c0, qlist);
    E1 = dense_graph_batch_calculate_E_from_qbits(h0, J0, c0, qlist, dtype);
    assert np.allclose(E0, E1)


    # rbm

    N0 = 4
    N1 = 3
    W = np.random.random((N1, N0))
    b0 = np.random.random((N0))
    b1 = np.random.random((N1))
    
    x0 = utils.generate_random_bits(N0)
    x1 = utils.generate_random_bits(N1)
    E0 = py.solver_traits.rbm_calculate_E(b0, b1, W, x0, x1)
    E1 = rbm_calculate_E(b0, b1, W, x0, x1, dtype)
    assert np.allclose(E0, E1)

    xlist0 = utils.create_bits_sequence(range(0, 1 << N0), N0)
    xlist1 = utils.create_bits_sequence(range(0, 1 << N1), N1)
    E0 = py.solver_traits.rbm_batch_calculate_E(b0, b1, W, xlist0, xlist1)
    E1 = rbm_batch_calculate_E(b0, b1, W, xlist0, xlist1, dtype)
    assert np.allclose(E0, E1)

    if False :
        h0, J0, c0 = py.solver_traits.dense_graph_calculate_hJc(W)
        h1, J1, c1 = dense_graph_calculate_hJc(W, dtype)
        assert np.allclose(h0, h1);
        assert np.allclose(J0, J1);
        assert np.allclose(c0, c1);

        q = utils.bits_to_qbits(x)
        E0 = py.solver_traits.dense_graph_calculate_E_from_qbits(h0, J0, c0, q);
        E1 = dense_graph_calculate_E_from_qbits(h0, J0, c0, q, dtype);
        assert np.allclose(E0, E1)

        qlist = utils.bits_to_qbits(xlist)
        E0 = py.solver_traits.dense_graph_batch_calculate_E_from_qbits(h0, J0, c0, qlist);
        E1 = dense_graph_batch_calculate_E_from_qbits(h0, J0, c0, qlist, dtype);
        assert np.allclose(E0, E1)
    

    #W = np.ones((3, 3))
    #h0, J0, c0 = py.solver_traits.dense_graph_calculate_hJc(W)
    #h1, J1, c1 = dense_graph_calculate_hJc(W, dtype)
        

    """
    print q
    print E0
    print E1
    """    
