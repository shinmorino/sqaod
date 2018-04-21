import numpy as np
from . import common

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
    tokens = [var, ' is not the bit type(np.int8). ', str(type(var))]
    msg = ''.join(tokens)
    raise Exception(msg)

def raise_wrong_shape(caption, mat) :
    tokens = ['wrong shape dim=(', caption, ') = ']
    tokens.append(str(mat.shape))
    msg = ''.join(tokens)
    raise Exception(msg)

def raise_not_symmetric(caption, mat) :
    tokens = ['given mat is not symmetric dim=(', caption, ') = ']
    tokens.append(str(mat.shape))
    msg = ''.join(tokens)
    raise Exception(msg)

def raise_dims_dont_match(caption, varlist) :
    tokens = ['dimension mismatch for ', caption, '. ']
    for var in varlist :
        tokens.append(str(var.shape))
        tokens.append(', ')
    tokens.pop()
    msg = ''.join(tokens)
    raise Exception(msg)

# value/type checks

def assert_is_bits(bitslist) :
    for bits in bitslist :
        if bits.dtype != np.int8 :
            raise Exception("bits are not given as np.int8")

def assert_is_vector(caption, vec) :
    if not common.is_vector(vec) :
        raise_not_a_vector(caption, vec)

def assert_is_scalar(caption, scalar) :
    if not common.is_scalar(scalar) :
        raise_not_a_scalar(caption, scalar)

# dense graph    

class dense_graph :

    @staticmethod
    def qubo(W) :
        if len(W.shape) != 2 or W.shape[0] != W.shape[1] :
            raise_wrong_shape('W', W)
        # W should be symmetric
        if not common.is_symmetric(W) :
            raise_not_symmetric("W", (W))

    # ising model dimension check    
    @staticmethod
    def hJc(h, J, c) :
        assert_is_vector('h', h)
        if len(J.shape) != 2 or J.shape[0] != J.shape[1] :
            raise_wrong_shape('J', J)
        if not common.is_symmetric(J) :
            raise_not_symmetric("J", (J))
        assert_is_scalar('c', (c))

    @staticmethod
    def bits(W, x) :
        assert_is_bits(x)
        matched = (W.shape[0] == x.shape[len(x.shape) - 1])
        if not matched :
            raise_dims_dont_match('W, x', (W, x))
        assert_is_bits((x))

# bipartite_graph

class bipartite_graph :
    @staticmethod
    def qubo(b0, b1, W) :
        assert_is_vector('b0', b0);
        assert_is_vector('b1', b1);
        if len(W.shape) != 2 :
            raise_wrong_shape('W', W)
        matched = (b0.shape[0] == W.shape[1]) and (b1.shape[0] == W.shape[0])
        if not matched :
            raise_dims_dont_match('b0, b1, W', (b0, b1, W))

    @staticmethod
    def hJc(h0, h1, J, c) :
        assert_is_vector('h0', h0);
        assert_is_vector('h1', h1);
        if len(J.shape) != 2 :
            raise_wrong_shape('J', J)
        assert_is_scalar('c', c)
        matched = (h0.shape[0] == J.shape[1]) and (h1.shape[0] == J.shape[0])
        if not matched :
            raise_dims_dont_match('h0, h1, J', (h0, h1, J))

    @staticmethod
    def bits(W, x0, x1) :
        assert_is_bits(x0)
        assert_is_bits(x1)
        # dimension checks
        matched = (W.shape[1] == x0.shape[len(x0.shape) - 1]) and \
                  (W.shape[0] == x1.shape[len(x1.shape) - 1])
        if len(x0.shape) != 1 or len(x1.shape) != 1 :
            matched &= len(x0.shape) == 2 and len(x1.shape) == 2
            matched &= x0.shape[0] == x1.shape[0]
            
        if not matched :
            raise_dims_dont_match('x0, x1', (x0, x1))
