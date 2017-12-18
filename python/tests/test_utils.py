import unittest
import numpy as np
import sqaod.common as common


class TestCommon(unittest.TestCase):
    
    def test_generate_W(self):
        N = 3
        W = common.generate_random_symmetric_W(N, dtype=np.float64)
        self.assertTrue(common.is_symmetric(W))

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
