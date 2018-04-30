from __future__ import print_function
from __future__ import absolute_import

import sqaod
import sqaod.py.formulas as py_formulas
import numpy as np
import unittest

class TestWrongType(unittest.TestCase) :

    def test_exception(self) :
        dtype = np.float64
        with self.assertRaises(RuntimeError):
            W = np.ones((4, 4), np.int32)
            # passing np.int32 which is an unexpected dtype.
            sqaod.cpu.formulas.dense_graph_calculate_hamiltonian(W, np.int32)


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
