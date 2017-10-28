import annealer
import numpy as np


qubo = np.array([[-32,4,4,4,4,4,4,4],
                 [4,-32,4,4,4,4,4,4],
                 [4,4,-32,4,4,4,4,4],
                 [4,4,4,-32,4,4,4,4],
                 [4,4,4,4,-32,4,4,4],
                 [4,4,4,4,4,-32,4,4],
                 [4,4,4,4,4,4,-32,4],
                 [4,4,4,4,4,4,4,-32]])


ann = annealer.simple_annealer(8, 4)
ann.set_qubo(qubo)

Ginit = 5.
Gfin = 0.01
m = 4

N=8
nRepeat = 4
kT = 0.02
tau = 0.99

for loop in range(0, nRepeat) :
    G = Ginit
    ann.randomize_q()
    while Gfin < G :
        ann.anneal_one_step(G, kT)
        G = G * tau

    q = ann.get_q() 
    E = ann.get_E() 
    print(q[0,:], E)
