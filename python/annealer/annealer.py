
import numpy as np
import random
import native


class simple_annealer :

    def __init__(use_native=True) :
        self.use_native=use_native
    
    def __init__(self, N, m, use_native=True) :
        self.set_problem_size(N, m)
        self.use_native=use_native

    def set_seed(seed) :
        if not self.use_native :
            random.seed(seed)
        else :
            native.set_seed(seed)
        
    def set_problem_size(self, N, m) :
        self.N = N
        self.m = m;
        self.q = np.zeros((m, N), dtype=np.int8)
    
    def set_qubo(self, qubo) :
        if not self.use_native :
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
                c = Jsum + hsum
        else :
            if qubo.dtype != 'float64' :
                qubo = qubo.astype(np.float64)
            h, J, c = native.create_hJc(qubo)

        self.h = h
        self.J = J
        self.c = c
            

    def get_q(self) :
        return self.q

    def get_hJc(self) :
        return h, J, c

    def get_E(self) :
        return self.E;
    
    def randomize_q(self) :
        if not self.use_native :
            for i in range(self.N):
                self.q.flat[i] = np.random.choice([-1,1])
        else :
            native.randomize_q(self.q)

    
    def anneal_one_step(self, G, kT) :
        q = self.q;
        h = self.h
        J = self.J
        c = self.c
        N = self.N
        m = self.m

        if not self.use_native :
            for i in range(self.N * self.m):
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

            self.E = E + c
        else :
            self.E = native.anneal_one_step(q, G, kT, h, J, c)
