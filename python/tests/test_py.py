import unittest
import numpy as np
from py import solver_traits
import utils
    

class TestTraits(unittest.TestCase):

    def setUp(self) :
        self.verbose = False

    # FIXME, add tests for batch version of energy calculation.
        
    def compare_energy(self, W, xlist) :
        Equbo = solver_traits.dense_graph_batch_calculate_E(W, xlist)
        qlist = utils.bits_to_qbits(xlist)
        h, J, c = solver_traits.dense_graph_calculate_hJc(W)
        EhJc = solver_traits.dense_graph_batch_calculate_E_from_qbits(h, J, c, qlist)

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
        xlist = utils.create_bits_sequence(range(0, 2 ** N), N)
        self.compare_energy(W, xlist)

    def test_engery_of_dense_graph_with_zero_x(self):
        N = 8
        W = utils.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = np.zeros(N, np.int8)
        Equbo = solver_traits.dense_graph_calculate_E(W, xlist)
        self.assertEqual(Equbo, 0.)

        self.compare_energy(W, xlist)

    def test_engery_of_dense_graph(self):
        N = 8
        W = utils.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = utils.create_bits_sequence(range(0, 2 ** N), N)
        self.compare_energy(W, xlist)

    def test_engery_of_batched_qubo_energy_func(self):
        N = 8
        W = utils.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = utils.create_bits_sequence(range(0, 2 ** N), N)

        E = []
        for i in range(0, 1 << N) :
            E.append(solver_traits.dense_graph_calculate_E(W, xlist[i]))

        Ebatch = solver_traits.dense_graph_batch_calculate_E(W, xlist)

        self.assertTrue(np.allclose(E, Ebatch))

    def test_engery_of_batched_qbits_energy_func(self):
        N = 8
        W = utils.generate_random_symmetric_W((N), dtype=np.float64)
        xlist = utils.create_bits_sequence(range(0, 2 ** N), N)
        qlist = utils.bits_to_qbits(xlist)
        h, J, c = solver_traits.dense_graph_calculate_hJc(W)
        
        E = []
        for i in range(0, 1 << N) :
            E.append(solver_traits.dense_graph_calculate_E_from_qbits(h, J, c, qlist[i]))

        Ebatch = solver_traits.dense_graph_batch_calculate_E_from_qbits(h, J, c, qlist)

        self.assertTrue(np.allclose(E, Ebatch))
        
        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
