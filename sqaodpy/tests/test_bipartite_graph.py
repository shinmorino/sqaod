from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from example_problems import *

class BipartiteGraphTestBase:

    def __init__(self, annealer_package) :
        self.annealer_package = annealer_package

    def test_min_energy(self):
        N0 = 200
        N1 = 350
        an = self.annealer_package.bipartite_graph_annealer()
        W = np.random.random((N1, N0)) - 0.5
        b0 = np.random.random((N0)) - 0.5
        b1 = np.random.random((N1)) - 0.5
        an.set_qubo(b0, b1, W)
        an.set_preferences(n_trotters = 1)
        an.prepare()

        q0 = np.ndarray((N0), np.int8)
        q1 = np.ndarray((N1), np.int8)
        q0[:] = -1
        q1[:] = -1
        an.set_q((q0, q1))
        an.calculate_E()
        self.assertTrue(np.allclose(an.get_E(), 0.))
        # print(an.E)

    def test_qubo_energy(self):
        N0 = 8
        N1 = 5
        an = self.annealer_package.bipartite_graph_annealer()
        W = np.random.random((N1, N0)) - 0.5
        b0 = np.random.random((N0)) - 0.5
        b1 = np.random.random((N1)) - 0.5
        an.set_qubo(b0, b1, W)
        an.set_preferences(n_trotters = 1)
        an.prepare()

        iMax = 1 << N0
        jMax = 1 << N1

        for i in range(iMax) :
            x0 = common.create_bitset_sequence((i,), N0)
            q0 = sq.bit_to_spin(x0)
            for j in range(jMax) :
                x1 = common.create_bitset_sequence((j,), N1)
                q1 = sq.bit_to_spin(x1)
                Ebf = np.dot(b0, x0.transpose()) + np.dot(b1, x1.transpose()) \
                      + np.dot(x1, np.matmul(W, x0.transpose()))
                an.set_q((q0, q1))
                an.calculate_E()
                Ean = an.get_E()
                if not np.allclose(Ebf, Ean) :
                    print(i, j, Ebf, Ean)
                self.assertTrue(np.allclose(Ebf,  Ean))

        
    def test_set_q(self):
        N0 = 100
        N1 = 50
        m = 150
        b0, b1, W = bipartite_graph_random(N0, N1, np.float32)
        an = self.annealer_package.bipartite_graph_annealer()
        an.set_qubo(b0, b1, W)
        an.set_preferences(n_trotters = m)
        an.prepare()
        
        q0in = 2 * np.random.randint(0, 2, N0) - 1
        q1in = 2 * np.random.randint(0, 2, N1) - 1
        an.set_q((q0in, q1in))
        qout = an.get_q()

        res = True
        for qpair in qout :
            res &= np.allclose(q0in, qpair[0]) and np.allclose(q1in, qpair[1])
        self.assertTrue(res)
            
    def test_set_q_m(self):
        N0 = 100
        N1 = 50
        m = 150
        b0, b1, W = bipartite_graph_random(N0, N1, np.float32)
        an = self.annealer_package.bipartite_graph_annealer()
        an.set_qubo(b0, b1, W)
        an.set_preferences(n_trotters = m)

        qin = []
        for loop in range(0, m) :
            q0 = 2 * np.random.randint(0, 2, N0) - 1
            q1 = 2 * np.random.randint(0, 2, N1) - 1
            qin.append((q0.astype(np.int8), q1.astype(np.int8)))
        an.set_q(qin)
        qout = an.get_q()

        self.assertTrue(len(qin) == len(qout))
        res = True
        for qinpair, qoutpair in zip(qin, qout) :
            res &= np.allclose(qinpair[0], qoutpair[0]) and np.allclose(qinpair[1], qoutpair[1])
        self.assertTrue(res)



class TestCPUBipartiteGraphAnnealer(BipartiteGraphTestBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        BipartiteGraphTestBase.__init__(self, sq.cpu)
        unittest.TestCase.__init__(self, testFunc)

class TestPyBipartiteGraphAnnealer(BipartiteGraphTestBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        BipartiteGraphTestBase.__init__(self, sq.py)
        unittest.TestCase.__init__(self, testFunc)

if sq.is_cuda_available() :
    import sqaod.cuda as sqcuda
    class TestCPUBipartiteGraphAnnealer(BipartiteGraphTestBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            BipartiteGraphTestBase.__init__(self, sqcuda)
            unittest.TestCase.__init__(self, testFunc)

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
