import unittest
import numpy as np
import rbm.rbm
from rbm.rbm_annealer import *
from rbm.rbm_bf_solver import *


class TestMinEnergy(unittest.TestCase):

    def test_min_energy(self):
        N0 = 200
        N1 = 350
        an = rbm_annealer(N0, N1, 1)
        W = np.random.random((N1, N0)) - 0.5
        b0 = np.random.random((N0)) - 0.5
        b1 = np.random.random((N1)) - 0.5
        an.set_qubo(W, b0, b1)

        q0 = np.ndarray((N0), np.int8)
        q1 = np.ndarray((N1), np.int8)
        q0[:] = -1
        q1[:] = -1
        an.set_q(0, q0)
        an.set_q(1, q1)
        an.calculate_E()
        self.assertTrue(an.E < 1.e-13)
        # print(an.E)


    def test_qubo_energy(self):
        N0 = 8
        N1 = 5
        an = rbm_annealer(N0, N1, 1)
        W = np.random.random((N1, N0)) - 0.5
        b0 = np.random.random((N0)) - 0.5
        b1 = np.random.random((N1)) - 0.5
        an.set_qubo(W, b0, b1)

        iMax = 1 << N0
        jMax = 1 << N1

        for i in range(iMax) :
            x0 = [np.int8(i >> pos & 1) for pos in range(N0 - 1,-1,-1)]
            an.set_x(0, x0)
            for j in range(jMax) :
                x1 = [np.int8(j >> pos & 1) for pos in range(N1 - 1,-1,-1)]
                Ebf = - np.dot(b0, x0) - np.dot(b1, x1) - np.dot(x1, np.matmul(W, x0))
                an.set_x(1, x1)
                Ean = an.calculate_E()
                self.assertTrue(np.absolute(Ebf - Ean) < 1.e-13)

    def test_bf_solver(self):
        N0 = 8
        N1 = 5

        an = rbm_annealer(N0, N1, 1)
        bf = rbm_bf_solver(N0, N1)
        
        W = np.random.random((N1, N0)) - 0.5
        b0 = np.random.random((N0)) - 0.5
        b1 = np.random.random((N1)) - 0.5

        an.set_qubo(W, b0, b1)
        bf.set_qubo(W, b0, b1)
        
        bf.search_optimum();
        rbm.rbm.anneal(an)

        # Assuming problems with small N0 and N1 give the same results
        # for annealer and brute-force solver.
        fbx0 = bf.get_x(0)
        fbx1 = bf.get_x(1)
        anx0 = an.get_x(0)
        anx1 = an.get_x(1)
        self.assertTrue(np.allclose(fbx0, anx0))
        self.assertTrue(np.allclose(fbx1, anx1))

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
