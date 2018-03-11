import unittest
import numpy as np
import sqaod.common as common
from sqaod.py import formulas
    

class TestTraits(unittest.TestCase):

    def setUp(self) :
        self.verbose = False

    # FIXME, add tests for batch version of energy calculation.
        
    def compare_energy(self, W, xlist) :
        Equbo = formulas.dense_graph_batch_calculate_E(W, xlist)
        qlist = common.bit_to_spin(xlist)
        h, J, c = formulas.dense_graph_calculate_hamiltonian(W)
        EhJc = formulas.dense_graph_batch_calculate_E_from_spin(h, J, c, qlist)

        if self.verbose :
            print 'xlist', xlist
            print 'qlist', qlist
            print 'E qubo \n', Equbo
            print 'E hJc  \n', EhJc
            print 'diff \n', np.absolute(Equbo - EhJc)
        self.assertTrue(np.allclose(Equbo, EhJc))
    
    def test_engery_of_known_W(self):
        # self.verbose = True
        N = 8
        W = np.ones((N, N), dtype=np.float64)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        self.compare_energy(W, xlist)

    def test_engery_of_dense_graph_with_zero_x(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = np.zeros(N, np.int8)
        Equbo = formulas.dense_graph_calculate_E(W, xlist)
        self.assertEqual(Equbo, 0.)

        self.compare_energy(W, xlist)

    def test_engery_of_dense_graph(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        self.compare_energy(W, xlist)

    def test_engery_of_batched_qubo_energy_func(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)

        E = []
        for i in range(0, 1 << N) :
            E.append(formulas.dense_graph_calculate_E(W, xlist[i]))

        Ebatch = formulas.dense_graph_batch_calculate_E(W, xlist)

        self.assertTrue(np.allclose(E, Ebatch))

    def test_engery_of_batched_spin_energy_func(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        qlist = common.bit_to_spin(xlist)
        h, J, c = formulas.dense_graph_calculate_hamiltonian(W)
        
        E = []
        for i in range(0, 1 << N) :
            E.append(formulas.dense_graph_calculate_E_from_spin(h, J, c, qlist[i]))

        Ebatch = formulas.dense_graph_batch_calculate_E_from_spin(h, J, c, qlist)

        self.assertTrue(np.allclose(E, Ebatch))
        
        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
