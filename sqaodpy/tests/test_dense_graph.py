from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from example_problems import *

class TestMethod(unittest.TestCase):
    
    def test_set_q(self):
        N = 100
        m = 100
        W = dense_graph_random(N, np.float32)
        
        an = sq.cpu.dense_graph_annealer()
        an.set_qubo(W)
        an.set_preferences(n_trotters = m)
        an.prepare()
        an.randomize_spin()
        
        qin = 2 * np.random.randint(0, 2, N) - 1
        an.set_q(qin)
        qout = an.get_q()

        for q in qout :
            self.assertTrue(np.allclose(qin, q))
            
    def test_set_q_m(self):
        N = 5
        m = 2
        W = dense_graph_random(N, np.float32)
        
        an = sq.cpu.dense_graph_annealer()
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

        self.assertTrue(len(qin) == len(qout))
        for q0, q1 in zip(qin, qout) :
            self.assertTrue(np.allclose(q0, q1))
            
    def test_set_q_m2(self):
        N = 5
        m = 2
        W = dense_graph_random(N, np.float32)
        
        an = sq.cpu.dense_graph_annealer()
        an.set_qubo(W)
        an.set_preferences(n_trotters = m)
        an.prepare()
        an.randomize_spin()

        qin = []
        for loop in range(0, m) :
            q = 2 * np.random.randint(0, 2, N) - 1
            qin.append(q.astype(np.int8))
        an.set_q(qin)
        #qout = an.get_q()

        #self.assertTrue(len(qin) == len(qout))
        #for q0, q1 in zip(qin, qout) :
        #    self.assertTrue(np.allclose(q0, q1))

        qin = 2 * np.random.randint(0, 2, N) - 1
        an.set_q(qin)
        qout = an.get_q()

        for q in qout :
            self.assertTrue(np.allclose(qin, q))


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
