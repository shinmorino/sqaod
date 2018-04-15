from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from example_problems import *

class TestMethod(unittest.TestCase):
    
    def test_set_q(self):
        N0 = 100
        N1 = 50
        m = 150
        b0, b1, W = bipartite_graph_random(N0, N1, np.float32)
        an = sq.cpu.bipartite_graph_annealer()
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
        an = sq.cpu.bipartite_graph_annealer()
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

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
