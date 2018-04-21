from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from .example_problems import *
from math import log
from math import exp

class TestBipartiteGraphBFSearcherBase:

    def __init__(self, anpkg, dtype) :
        self.anpkg = anpkg
        self.dtype = dtype
        self.epu = 4.e-5 if dtype == np.float32 else 1.e-8

    def new_searcher(self, N0, N1) :
        searcher = self.anpkg.bipartite_graph_bf_searcher(dtype=self.dtype)
        b0, b1, W = bipartite_graph_random(N0, N1, self.dtype)
        searcher.set_qubo(b0, b1, W)
        return searcher

    def test_calling_sequence(self) :
        N0, N1 = 8, 8
        searcher = self.new_searcher(N0, N1)
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
        N0, N1 = 8, 8
        searcher = self.new_searcher(N0, N1)
        N0out, N1out = searcher.get_problem_size()
        self.assertEqual(N0, N0out)
        self.assertEqual(N1, N1out)
        
    def _test_search(self, opt, b0, b1, W, Eexp, xexp):
        N0, N1 = b0.shape[0], b1.shape[0]
        searcher = self.new_searcher(N0, N1)
        searcher.set_qubo(b0, b1, W, opt)
        searcher.search()
        searcher.calculate_E()
        E = searcher.get_E()

        res = np.allclose(E[0], Eexp, atol=self.epu)
        #print(E[0], Eexp)
        self.assertTrue(res)

        searcher.make_solution()
        xlist = searcher.get_x()
        self.assertEqual(len(xlist), 1)
        x0, x1 = xlist[0]
        #print(x0, x1)
        self.assertTrue(np.allclose(x0, xexp))
        self.assertTrue(np.allclose(x1, xexp))
        
    def test_min_energy_positive_W(self):
        N0, N1 = 8, 8
        b0 = np.ones((N0), np.int8)
        b1 = np.ones((N1), np.int8)
        W = np.ones((N1, N0), np.int8)
        self._test_search(sq.minimize, b0, b1, W, 0, 0)
        
    def test_min_energy_negative_W(self):
        N0, N1 = 8, 8
        b0 = - np.ones((N0), np.int8)
        b1 = - np.ones((N1), np.int8)
        W = - np.ones((N1, N0), np.int8)
        #print(b0, b1, W)
        self._test_search(sq.minimize, b0, b1, W, np.sum(b0) + np.sum(b1) + np.sum(W), 1)
        
    def test_max_energy_positive_W(self):
        N0, N1 = 8, 8
        b0 = np.ones((N0), np.int8)
        b1 = np.ones((N1), np.int8)
        W = np.ones((N1, N0), np.int8)
        self._test_search(sq.maximize, b0, b1, W, np.sum(b0) + np.sum(b1) + np.sum(W), 1)
        
    def test_min_energy_negative_W(self):
        N0, N1 = 8, 8
        b0 = - np.ones((N0), np.int8)
        b1 = - np.ones((N1), np.int8)
        W = - np.ones((N1, N0), np.int8)
        #print(b0, b1, W)
        self._test_search(sq.maximize, b0, b1, W, 0, 0)

    #def test_max_energy(self):
        

class TestNativeBipartiteGraphBFSearcherBase(TestBipartiteGraphBFSearcherBase) :
    def __init__(self, anpkg, dtype) :
        TestBipartiteGraphBFSearcherBase.__init__(self, anpkg, dtype)

    def test_set_algorithm(self) :
        N0, N1 = 8, 8
        searcher = self.new_searcher(N0, N1)
        searcher.set_preferences(algorithm = sq.algorithm.default)
        pref = searcher.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.brute_force_search)

        searcher.set_preferences(algorithm = sq.algorithm.naive)
        pref = searcher.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.brute_force_search)

        searcher.set_preferences(algorithm = sq.algorithm.coloring)
        pref = searcher.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.brute_force_search)

    def test_prec_prefs(self) :
        searcher = self.new_searcher(8, 8)
        prefs = searcher.get_preferences()
        precstr = 'float' if self.dtype == np.float32 else 'double'
        self.assertEqual(prefs['precision'], precstr)

    def test_precision(self) :
        searcher = self.new_searcher(8, 8)
        self.assertEqual(searcher.dtype, self.dtype)


class TestCPUBipartiteGraphBFSearcherBase(TestNativeBipartiteGraphBFSearcherBase) :
    def __init__(self, dtype) :
        TestNativeBipartiteGraphBFSearcherBase.__init__(self, sq.cpu, dtype)

    def test_device_pref(self) :
        searcher = self.new_searcher(8, 8)
        prefs = searcher.get_preferences()
        self.assertEqual(prefs['device'], 'cpu')
        

class TestPyBipartiteGraphBFSearcher(TestBipartiteGraphBFSearcherBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestBipartiteGraphBFSearcherBase.__init__(self, sq.py, np.float64)
        unittest.TestCase.__init__(self, testFunc)


class TestCPUBipartiteGraphBFSearcherFP32(TestCPUBipartiteGraphBFSearcherBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUBipartiteGraphBFSearcherBase.__init__(self, np.float32)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUBipartiteGraphBFSearcherFP64(TestCPUBipartiteGraphBFSearcherBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUBipartiteGraphBFSearcherBase.__init__(self, np.float64)
        unittest.TestCase.__init__(self, testFunc)


if sq.is_cuda_available() :
    class TestCUDABipartiteGraphBFSearcherBase(TestNativeBipartiteGraphBFSearcherBase) :
        def __init__(self, dtype) :
            TestNativeBipartiteGraphBFSearcherBase.__init__(self, sq.cuda, dtype)

        def test_device_pref(self) :
            searcher = self.new_searcher(8, 8)
            prefs = searcher.get_preferences()
            self.assertEqual(prefs['device'], 'cuda')
            
    class TestCUDABipartiteGraphBFSearcherFP32(TestCUDABipartiteGraphBFSearcherBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDABipartiteGraphBFSearcherBase.__init__(self, np.float32)
            unittest.TestCase.__init__(self, testFunc)
            

    class TestCUDABipartiteGraphBFSearcherFP64(TestCUDABipartiteGraphBFSearcherBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDABipartiteGraphBFSearcherBase.__init__(self, np.float64)
            unittest.TestCase.__init__(self, testFunc)

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
