import numpy as np


def create_bits_sequence(vals, nbits) :
    if isinstance(vals, list) or isinstance(vals, tuple) :
        seqlen = len(vals)
        x = np.ndarray((seqlen, nbits), np.int8)
        iseq = 0
        for v in vals :
            for pos in range(nbits - 1, -1, -1) :
                x[iseq][pos] = np.int8(v >> pos & 1)
            iseq += 1
        return x
    vals = np.int32(vals)
    return create_bits_sequence(range(vals, vals + 1), nbits)


def randomize_qbits(qbits) :
    for qbit in np.nditer(qbits, [], [['readwrite']]) :
        qbit[...] = np.random.choice([-1,1])

def x_to_q(x) :
    q = x.copy() * 2 - 1
    return q
