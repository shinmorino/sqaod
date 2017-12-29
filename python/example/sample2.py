import annealer
import numpy as np


Ginit = 5.
Gfin = 0.01
N = 16
m = 8

nRepeat = 4
kT = 0.02
tau = 0.99


def anneal(ann) :

    qubo = np.zeros((N, N))
    for icol in range(0, N) :
        for irow in range(0, N) :
            qubo[icol][irow] = 4
    for i in range(0, N) :
        qubo[i][i] = -32
    

    ann.set_qubo(qubo)

    for loop in range(0, nRepeat) :
        G = Ginit
        ann.randomize_q()
        while Gfin < G :
            ann.anneal_one_step(G, kT)
            G = G * tau

        q = ann.get_q() 
        E = ann.get_E() 
        print(q[0,:], E)


print('simple annealer (python)')
anneal(annealer.simple_annealer(N, m))
print
print('simple native annealer (native)')
anneal(annealer.simple_native_annealer(N, m))
