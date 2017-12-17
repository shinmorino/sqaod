import unittest
import numpy as np
import sqaod.utils as utils
from sqaod.py import solver_traits


class TestUtils(unittest.TestCase):
    
    def test_generate_W(self):
        N = 3
        W = utils.generate_random_symmetric_W(N, dtype=np.float64)
        self.assertTrue(utils.is_symmetric(W))

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
