import unittest
import numpy as np
from py import dense_graph_traits
import utils
    

class TestTraits(unittest.TestCase):

    def setUp(self) :
        self.verbose = False
        
    def compare_energy(self, W, xlist) :
        Equbo = dense_graph_traits.calculate_qubo_E(W, xlist)
        qlist = utils.x_to_q(xlist)
        h, J, c = dense_graph_traits.calculate_hJc(W)
        EhJc = dense_graph_traits.calculate_E_from_hJc(h, J, c, qlist)

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

    def test_engery_of_zero_x(self):
        N = 8
        W = dense_graph_traits.generate_random_W((N), dtype=np.float64)
        xlist = np.zeros(N, np.int8)
        self.compare_energy(W, xlist)
        

    def test_engery_of_dense_graph(self):
        N = 8
        W = dense_graph_traits.generate_random_W((N), dtype=np.float64)
        xlist = utils.create_bits_sequence(range(0, 2 ** N), N)
        self.compare_energy(W, xlist)

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
