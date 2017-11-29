import unittest
import numpy as np
import utils
from py import dense_graph_traits


class TestUtils(unittest.TestCase):
    
    def test_generate_W(self):
        N = 3
        W = dense_graph_traits.generate_random_W(N, dtype=np.float64)
        self.assertTrue(dense_graph_traits.is_symmetric(W))

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
