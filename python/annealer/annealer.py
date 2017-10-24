import numpy as np
import random

def createJhc(qubo) :
    N = len(qubo)
    h = np.zeros(N, dtype=np.float64)
    J = np.zeros((N, N), dtype=np.float64)
    Jsum = 0.
    hsum = 0.

    for j in range(N):
        sum = 0
        
        for i in range(j+1,N):
            r = qubo[j][i]
            sum += r
            J[j][i] = r*1.0/4
            J[i][j] = r*1.0/4
            Jsum += r*1.0/4
            
        for i in range(0, j):
            sum += qubo[j][i]
            
        s = qubo[j][j]
        hsum += s * 1.0 / 2
        h[j] = s * 1.0 / 2 + sum

    return (J, h, Jsum + hsum)
    
    
def createQ(N, m) :
    return np.zeros((m, N), dtype=np.int8) 

def randomizeQ(q) :
    for i in range(q.size):
	q.flat[i] = np.random.choice([-1,1])

                    
def anneal(q, G, kT, m, J, h, c) :
    N = q.shape[1]
    for i in range(N * m):
	x = np.random.randint(N)
	y = np.random.randint(m)
	dE = (2*q[y][x]*(h[x]+q[y][(N+x-1)%N]*J[x][(N+x-1)%N]+q[y][(x+1)%N]*J[x][(x+1)%N]))*1.0/m
	dE += -q[y][x]*(q[(m+y-1)%m][x]+q[(y+1)%m][x])*np.log(np.tanh(G/kT/m))*1.0/kT
	if np.exp(-dE/kT)> np.random.rand():
	    q[y][x] = -q[y][x]
            
    E = 0
    for a in range(N):
	E += h[a]*q[0][a]
	for b in range(a+1,N):
	    E += J[a][b]*q[0][a]*q[0][b]

    return E + c
