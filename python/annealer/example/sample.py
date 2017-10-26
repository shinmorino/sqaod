import annealer
import numpy as np


qubo = np.array([[-32,4,4,4,4,4,4,4],
                 [4,-32,4,4,4,4,4,4],
                 [4,4,-32,4,4,4,4,4],
                 [4,4,4,-32,4,4,4,4],
                 [4,4,4,4,-32,4,4,4],
                 [4,4,4,4,4,-32,4,4],
                 [4,4,4,4,4,4,-32,4],
                 [4,4,4,4,4,4,4,-32]],
                np.float64)

h, J, c = annealer.create_hJc(qubo)

Ginit = 5.
Gfin = 0.01
m = 4

N=8
nRepeat = 4
kT = 0.02
tau = 0.99

q = annealer.create_q(N, m)
Ec = 0
for loop in range(0, nRepeat) :
    G = Ginit
    annealer.randomize_q(q)
    while Gfin < G :
        Ec = annealer.anneal_one_step(q, G, kT, h, J, c)
        G = G * tau

    print(q[0,:], Ec)
