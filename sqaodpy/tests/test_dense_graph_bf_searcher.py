from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from example_problems import *
from math import log, exp

class TestDenseGraphBFSearcherBase:

    def __init__(self, anpkg, dtype) :
        self.anpkg = anpkg
        self.dtype = dtype
        self.epu = 4.e-5 if dtype == np.float32 else 1.e-8

    def new_searcher(self, N) :
        searcher = self.anpkg.dense_graph_bf_searcher(dtype=self.dtype)
        W = dense_graph_random(N, self.dtype)
        searcher.set_qubo(W)
        return searcher

    def test_calling_sequence(self) :
        N = 10
        searcher = self.new_searcher(N)
        searcher.prepare()
        searcher.search_range()
        searcher.make_solution()
        searcher.search()
        
        searcher.calculate_E()
        searcher.get_E()
        searcher.get_problem_size()
        searcher.get_preferences()
        x = searcher.get_x()

    def test_problem_size(self) :
        N = 100
        searcher = self.new_searcher(N)
        Nout = searcher.get_problem_size()
        self.assertEqual(N, Nout)
        
    def _test_search(self, opt, W, Eexp, xexp):
        N = W.shape[0]
        searcher = self.new_searcher(N)
        searcher.set_qubo(W, opt)
        searcher.prepare()
        searcher.search()

        searcher.calculate_E()
        E = searcher.get_E()
        self.assertTrue(np.allclose(E[0], Eexp, atol=self.epu))

        searcher.make_solution()
        x = searcher.get_x()
        self.assertTrue(np.allclose(x, xexp))
        
    def test_min_energy_positive_W(self):
        N = 8
        W = np.ones((N, N), np.int8)
        self._test_search(sq.minimize, W, 0., 0)

    def test_min_energy_negative_W(self):
        N = 8
        W = - np.ones((N, N), np.int8)
        self._test_search(sq.minimize, W, np.sum(W), 1)
        
    def test_max_energy_positive_W(self):
        N = 8
        W = np.ones((N, N), np.int8)
        self._test_search(sq.maximize, W, np.sum(W), 1)

    def test_max_energy_negative_W(self):
        N = 8
        W = - np.ones((N, N), np.int8)
        self._test_search(sq.maximize, W, 0., 0)
        

class TestNativeDenseGraphBFSearcherBase(TestDenseGraphBFSearcherBase) :
    def __init__(self, anpkg, dtype) :
        TestDenseGraphBFSearcherBase.__init__(self, anpkg, dtype)

    def test_set_algorithm(self) :
        ann = self.new_searcher(10)
        ann.set_preferences(algorithm = sq.algorithm.default)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.brute_force_search)

        ann.set_preferences(algorithm = sq.algorithm.naive)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.brute_force_search)

        ann.set_preferences(algorithm = sq.algorithm.coloring)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.brute_force_search)

    def test_prec_prefs(self) :
        searcher = self.new_searcher(10)
        prefs = searcher.get_preferences()
        precstr = 'float' if self.dtype == np.float32 else 'double'
        self.assertEqual(prefs['precision'], precstr)

    def test_precision(self) :
        searcher = self.new_searcher(10)
        self.assertEqual(searcher.dtype, self.dtype)


class TestCPUDenseGraphBFSearcherBase(TestNativeDenseGraphBFSearcherBase) :
    def __init__(self, dtype) :
        TestNativeDenseGraphBFSearcherBase.__init__(self, sq.cpu, dtype)

    def test_device_pref(self) :
        searcher = self.new_searcher(10)
        prefs = searcher.get_preferences()
        self.assertEqual(prefs['device'], 'cpu')
        

class TestPyDenseGraphBFSearcher(TestDenseGraphBFSearcherBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestDenseGraphBFSearcherBase.__init__(self, sq.py, np.float64)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUDenseGraphBFSearcherFP32(TestCPUDenseGraphBFSearcherBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUDenseGraphBFSearcherBase.__init__(self, np.float32)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUDenseGraphBFSearcherFP64(TestCPUDenseGraphBFSearcherBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUDenseGraphBFSearcherBase.__init__(self, np.float64)
        unittest.TestCase.__init__(self, testFunc)


if sq.is_cuda_available() :
    class TestCUDADenseGraphBFSearcherBase(TestNativeDenseGraphBFSearcherBase) :
        def __init__(self, dtype) :
            TestNativeDenseGraphBFSearcherBase.__init__(self, sq.cuda, dtype)

        def test_device_pref(self) :
            searcher = self.new_searcher(10)
            prefs = searcher.get_preferences()
            self.assertEqual(prefs['device'], 'cuda')
            
    class TestCUDADenseGraphBFSearcherFP32(TestCUDADenseGraphBFSearcherBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDADenseGraphBFSearcherBase.__init__(self, np.float32)
            unittest.TestCase.__init__(self, testFunc)
            

    class TestCUDADenseGraphBFSearcherFP64(TestCUDADenseGraphBFSearcherBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDADenseGraphBFSearcherBase.__init__(self, np.float64)
            unittest.TestCase.__init__(self, testFunc)

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
