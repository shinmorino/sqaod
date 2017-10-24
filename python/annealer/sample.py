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

J, h, c = annealer.createJhc(qubo)

Ginit = 5.
Gfin = 0.01
m = 4

N=8
nRepeat = 4
kT = 0.02
tau = 0.99

q = annealer.createQ(N, m)
Ec = 0
for loop in range(0, nRepeat) :
    G = Ginit
    annealer.randomizeQ(q)
    while Gfin < G :
        Ec = annealer.anneal(q, G, kT, m, J, h, c)
        G = G * tau

    print(q[0,:], Ec)
