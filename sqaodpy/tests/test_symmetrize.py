import numpy as np
import sqaod as sq
from tests import example_problems
import unittest


class Base :
    def __init__(self) :
        assert False, "Not intended to be called."
        
    class TestSymmetrize(unittest.TestCase) :

        N = 1024

        def test_is_triangular_triu(self) :
            W = example_problems.dense_graph_random(self.N, self.dtype)
            self.assertTrue(sq.common.is_triangular(np.triu(W)))

        def test_is_triangular_tril(self) :
            W = example_problems.dense_graph_random(self.N, self.dtype)
            self.assertTrue(sq.common.is_triangular(np.tril(W)))

        def test_is_not_triangular(self) :
            W = np.ones((self.N, self.N), self.dtype)
            self.assertFalse(sq.common.is_triangular(W))

        def test_symmetrize_triu(self) :
            W = example_problems.upper_triangular_random_matrix(self.N, self.dtype)
            self.assertFalse(sq.common.is_symmetric(W))
            Wsym = sq.common.symmetrize(W)
            self.assertTrue(sq.common.is_symmetric(Wsym))
            self.assertTrue(np.allclose(W, sq.common.symmetric_to_triu(Wsym)))
            self.assertNotEqual(id(W), id(Wsym))

        def test_symmetrize_tril(self) :
            W = example_problems.lower_triangular_random_matrix(self.N, self.dtype)
            self.assertFalse(sq.common.is_symmetric(W))
            Wsym = sq.common.symmetrize(W)
            self.assertTrue(sq.common.is_symmetric(Wsym))
            self.assertTrue(np.allclose(W, sq.common.symmetric_to_tril(Wsym)))
            self.assertNotEqual(id(W), id(Wsym))

        def test_symmetrize_symmetric(self) :
            W = example_problems.dense_graph_random(self.N, self.dtype)
            self.assertTrue(sq.common.is_symmetric(W))
            Wsym = sq.common.symmetrize(W)
            self.assertTrue(sq.common.is_symmetric(Wsym))
            self.assertTrue(np.allclose(W, Wsym))
            self.assertEqual(id(W), id(Wsym))

        def test_symmetrize_invalid(self) :
            W = np.asarray(np.random.random((self.N, self.N)), self.dtype)
            with self.assertRaises(RuntimeError) :
                sq.common.symmetrize(W)
            
            

class TestSymmetrize64(Base.TestSymmetrize) :
    dtype = np.float64

class TestSymmetrize32(Base.TestSymmetrize) :
    dtype = np.float32
            
        
        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
