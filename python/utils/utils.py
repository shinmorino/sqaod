import sys
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



def anneal(annealer, Ginit = 5., Gfin = 0.01, kT = 0.02, tau = 0.99, n_repeat = 10, verbose = False) :
    Emin = sys.float_info.max
    q0 = []
    q1 = []
    
    for loop in range(0, n_repeat) :
        annealer.randomize_q(0)
        annealer.randomize_q(1)
        G = Ginit
        while Gfin < G :
            annealer.anneal_one_step(G, kT)
            if verbose :
                E - annealer.calculate_E()
                print E
            G = G * tau

        E = annealer.calculate_E()
        if E < Emin :
            q0 = annealer.get_q(0)[0] 
            q1 = annealer.get_q(1)[0]
            Emin = E

    return E, q0, q1
