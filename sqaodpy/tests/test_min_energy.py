from __future__ import print_function
import unittest
import numpy as np
import sqaod.py as py
import sqaod.common as common


class TestMinEnergy(unittest.TestCase):

    def test_bf_searcher(self):
        N0 = 4
        N1 = 4

        an = py.bipartite_graph_annealer()
        bf = py.bipartite_graph_bf_searcher()
        
        W = np.random.random((N1, N0)) - 0.5
        b0 = np.random.random((N0)) - 0.5
        b1 = np.random.random((N1)) - 0.5

        an.set_qubo(b0, b1, W)
        bf.set_qubo(b0, b1, W)
        
        bf.search();
        common.anneal(an)

        # Assuming problems with small N0 and N1 give the same results
        # for annealer and brute-force solver.
        Ebf = bf.get_E()[0]
        bfx0, bfx1 = bf.get_x()[0]
        Ean = an.get_E()[0]
        anx0, anx1 = an.get_x()[0]
        if not np.allclose(bfx0, anx0) or not np.allclose(bfx1, anx1) :
            print(bfx0, anx0, an.get_E())
            print(bfx1, anx1, bf.get_E())
        self.assertTrue(np.allclose(bfx0, anx0))
        self.assertTrue(np.allclose(bfx1, anx1))

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
