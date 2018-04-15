from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from example_problems import *

class DenseGraphTestBase:

    def __init__(self, annealer_package) :
        self.annealer_package = annealer_package
    
    def test_min_energy(self):
        N = 200
        W = dense_graph_random(N, np.float32)

        an = self.annealer_package.dense_graph_annealer()
        an.set_qubo(W)
        an.set_preferences(n_trotters = 1)
        an.prepare()
        
        q = np.ndarray((N), np.int8)
        q[:] = -1
        an.set_q(q)
        an.calculate_E()
        self.assertTrue(np.allclose(an.get_E(), 0.))
        # print(an.E)

    def test_set_q(self):
        N = 100
        m = 100
        W = dense_graph_random(N, np.float32)
        
        an = self.annealer_package.dense_graph_annealer()
        an.set_qubo(W)
        an.set_preferences(n_trotters = m)
        an.prepare()
        an.randomize_spin()
        
        qin = 2 * np.random.randint(0, 2, N) - 1
        an.set_q(qin)
        qout = an.get_q()

        res = True;
        for q in qout :
            res &= np.allclose(qin, q)
        self.assertTrue(res)
            
    def test_set_q_m(self):
        N = 5
        m = 2
        W = dense_graph_random(N, np.float32)
        
        an = self.annealer_package.dense_graph_annealer()
        an.set_qubo(W)
        an.set_preferences(n_trotters = m)
        an.prepare()
        an.randomize_spin()

        qin = []
        for loop in range(0, m) :
            q = 2 * np.random.randint(0, 2, N) - 1
            qin.append(q.astype(np.int8))
        an.set_q(qin)
        qout = an.get_q()

        res= len(qin) == len(qout)
        if res :
            for q0, q1 in zip(qin, qout) :
                res &= np.allclose(q0, q1)
        self.assertTrue(res)
        

class TestCPUDenseGraphAnnealer(DenseGraphTestBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        DenseGraphTestBase.__init__(self, sq.cpu)
        unittest.TestCase.__init__(self, testFunc)

class TestPyDenseGraphAnnealer(DenseGraphTestBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        DenseGraphTestBase.__init__(self, sq.py)
        unittest.TestCase.__init__(self, testFunc)

if sq.is_cuda_available() :
    import sqaod.cuda as sqcuda
    class TestCPUDenseGraphAnnealer(DenseGraphTestBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            DenseGraphTestBase.__init__(self, sqcuda)
            unittest.TestCase.__init__(self, testFunc)

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
