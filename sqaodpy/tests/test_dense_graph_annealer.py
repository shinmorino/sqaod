from __future__ import print_function
from __future__ import absolute_import
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from .example_problems import *
from math import log
from math import exp

class TestDenseGraphAnnealerBase:

    def __init__(self, anpkg, dtype) :
        self.anpkg = anpkg
        self.dtype = dtype
        self.epu = 1.e-6 if dtype == np.float32 else 1.e-12

    def new_annealer(self, N, m) :
        an = self.anpkg.dense_graph_annealer(dtype=self.dtype)
        W = dense_graph_random(N, self.dtype)
        an.set_qubo(W)
        an.set_preferences(n_trotters = m)
        return an

    def test_calling_sequence(self) :
        N = 200
        m = 100
        an = self.new_annealer(N, m)
        an.set_preferences(algorithm = sq.algorithm.default)
        an.seed(0)
        an.prepare()
        an.randomize_spin()
        an.anneal_one_step(1, 1)
        an.calculate_E()
        an.make_solution()
        an.get_E()
        an.get_problem_size()
        an.get_preferences()
        x = an.get_x()
        h, J, c = an.get_hamiltonian()
        q = an.get_q()

    def test_problem_size(self) :
        N = 100
        m = 10
        an = self.new_annealer(N, m)
        Nout = an.get_problem_size()
        self.assertEqual(N, Nout)
        
    def test_set_n_trotters(self) :
        an = self.new_annealer(10, 10)
        an.set_preferences(n_trotters = 3)
        prefs = an.get_preferences()
        self.assertEqual(prefs['n_trotters'], 3)

    def test_get_hamiltonian(self) :
        N = 100
        m = 10
        an = self.anpkg.dense_graph_annealer(dtype=self.dtype)
        W = dense_graph_random(N, self.dtype)
        an.set_qubo(W)
        h0, J0, c0 = an.get_hamiltonian()
        h1, J1, c1 = sq.py.formulas.dense_graph_calculate_hamiltonian(W)
        self.assertTrue(np.allclose(h0, h1, atol=self.epu))
        self.assertTrue(np.allclose(J0, J1, atol=self.epu))
        self.assertTrue(np.allclose(c0, c1, atol=self.epu))

    def test_set_hamiltonian(self) :
        N = 100
        m = 10
        W = dense_graph_random(N, self.dtype)
        h0, J0, c0 = sq.py.formulas.dense_graph_calculate_hamiltonian(W)
        
        an = self.anpkg.dense_graph_annealer(dtype=self.dtype)
        an.set_hamiltonian(h0, J0, c0)
        an.prepare()
        
        h1, J1, c1 = an.get_hamiltonian()
        #print(h0, h1)
        self.assertTrue(np.allclose(h0, h1, atol=self.epu))
        self.assertTrue(np.allclose(J0, J1))
        self.assertTrue(np.allclose(c0, c1)) #, atol=self.epu))

        
    def test_min_energy(self):
        N = 200
        an = self.new_annealer(N, 1)
        an.prepare()
        
        q = np.ndarray((N), np.int8)
        q[:] = -1
        an.set_q(q)
        an.calculate_E()
        E = an.get_E()
        res = np.allclose(E[0], 0., atol=self.epu)
        self.assertTrue(res)

    def test_qubo_energy(self):
        N = 8
        an = self.new_annealer(N, 1)
        W = dense_graph_random(N, self.dtype)
        an.set_qubo(W)
        an.set_preferences(n_trotters = 1)
        an.prepare()

        nMax = 1 << N
        res = True
        for i in range(nMax) :
            x = common.create_bitset_sequence((i,), N)
            Eref = sq.py.formulas.dense_graph_calculate_E(W, x)
            
            q = sq.bit_to_spin(x)
            an.set_q(q)
            an.calculate_E()
            Ean = an.get_E()
            if not np.allclose(Eref,  Ean, atol=self.epu) :
                print(Eref, Ean, Eref - Ean)
            res &= np.allclose(Eref,  Ean, atol=self.epu)

        self.assertTrue(res)

    def test_set_q(self):
        N = 100
        m = 100
        an = self.new_annealer(N, m)
        an.prepare()
        
        qin = 2 * np.random.randint(0, 2, N) - 1
        an.set_q(qin)
        qout = an.get_q()

        res = True;
        for q in qout :
            res &= np.allclose(qin, q)
        self.assertTrue(res)

    def test_set_qset(self):
        N = 5
        m = 2
        an = self.new_annealer(N, m)
        qsetin = []
        for loop in range(0, m) :
            q = 2 * np.random.randint(0, 2, N) - 1
            qsetin.append(q.astype(np.int8))
        an.set_qset(qsetin)
        qout = an.get_q()

        res= len(qsetin) == len(qout)
        if res :
            for q0, q1 in zip(qsetin, qout) :
                res &= np.allclose(q0, q1)
        self.assertTrue(res)

    def anneal(self, an) :
        an.prepare()
        an.randomize_spin()

        Ginit, Gfin = 5, 0.01
        beta = 1. / 0.02
        nSteps = 100

        G = Ginit
        tau = exp(log(Gfin / Ginit) / nSteps)
        for loop in range(0, nSteps) :
            an.anneal_one_step(G, beta)
            G *= tau
        an.make_solution()

    def _test_anneal_minimize(self, algorithm) :
        N = 10
        m = 4
        an = self.anpkg.dense_graph_annealer(dtype=self.dtype)
        W = np.ones((N, N), dtype=self.dtype)
        an.set_qubo(W, sq.minimize)
        an.set_preferences(n_trotters = m)
        an.set_preferences(algorithm = algorithm)

        self.anneal(an)
        
        E = an.get_E()
        self.assertEqual(E.min(), 0)

    def _test_anneal_maximize(self, algorithm) :
        N = 10
        m = 4
        an = self.anpkg.dense_graph_annealer(dtype=self.dtype)
        W = - np.ones((N, N), dtype=self.dtype)
        an.set_qubo(W, sq.maximize)
        an.set_preferences(n_trotters = m)
        an.set_preferences(algorithm = algorithm)

        self.anneal(an)

        E = an.get_E()
        self.assertEqual(E.max(), 0)

    def _test_anneal_hamiltonian(self, algorithm) :
        N = 10
        m = 4
        an = self.anpkg.dense_graph_annealer(dtype=self.dtype)
        W = np.ones((N, N), dtype=self.dtype)
        h, J, c = sq.py.formulas.dense_graph_calculate_hamiltonian(W)
        an.set_hamiltonian(h, J, c)
        an.set_preferences(n_trotters = m)
        an.set_preferences(algorithm = algorithm)

        self.anneal(an)

        E = an.get_E()
        self.assertEqual(E.max(), 0)

    def test_anneal_minimize(self) :
        self._test_anneal_minimize(sq.algorithm.naive)
        self._test_anneal_minimize(sq.algorithm.coloring)
        self._test_anneal_minimize(sq.algorithm.default)

    def test_anneal_maximize(self) :
        self._test_anneal_maximize(sq.algorithm.naive)
        self._test_anneal_maximize(sq.algorithm.coloring)
        self._test_anneal_maximize(sq.algorithm.default)

    def test_anneal_hamiltonian(self) :
        self._test_anneal_hamiltonian(sq.algorithm.naive)
        self._test_anneal_hamiltonian(sq.algorithm.coloring)
        self._test_anneal_hamiltonian(sq.algorithm.default)
        

class TestNativeDenseGraphAnnealerBase(TestDenseGraphAnnealerBase) :
    def __init__(self, anpkg, dtype) :
        TestDenseGraphAnnealerBase.__init__(self, anpkg, dtype)

    def test_set_algorithm(self) :
        ann = self.new_annealer(10, 10)
        ann.set_preferences(algorithm = sq.algorithm.default)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

        ann.set_preferences(algorithm = sq.algorithm.naive)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.naive)

        ann.set_preferences(algorithm = sq.algorithm.coloring)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

    def test_prec_prefs(self) :
        an = self.new_annealer(10, 10)
        prefs = an.get_preferences()
        precstr = 'float' if self.dtype == np.float32 else 'double'
        self.assertEqual(prefs['precision'], precstr)

    def test_precision(self) :
        an = self.new_annealer(10, 10)
        self.assertEqual(an.dtype, self.dtype)


class TestCPUDenseGraphAnnealerBase(TestNativeDenseGraphAnnealerBase) :
    def __init__(self, dtype) :
        TestNativeDenseGraphAnnealerBase.__init__(self, sq.cpu, dtype)

    def test_device_pref(self) :
        an = self.new_annealer(10, 10)
        prefs = an.get_preferences()
        self.assertEqual(prefs['device'], 'cpu')

    def test_set_algorithm(self) :
        ann = self.new_annealer(10, 10)
        ann.set_preferences(algorithm = sq.algorithm.default)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

        ann.set_preferences(algorithm = sq.algorithm.naive)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.naive)

        ann.set_preferences(algorithm = sq.algorithm.coloring)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)


class TestPyDenseGraphAnnealer(TestDenseGraphAnnealerBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestDenseGraphAnnealerBase.__init__(self, sq.py, np.float64)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUDenseGraphAnnealerFP32(TestCPUDenseGraphAnnealerBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUDenseGraphAnnealerBase.__init__(self, np.float32)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUDenseGraphAnnealerFP64(TestCPUDenseGraphAnnealerBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUDenseGraphAnnealerBase.__init__(self, np.float64)
        unittest.TestCase.__init__(self, testFunc)


if sq.is_cuda_available() :
    class TestCUDADenseGraphAnnealerBase(TestNativeDenseGraphAnnealerBase) :
        def __init__(self, dtype) :
            TestNativeDenseGraphAnnealerBase.__init__(self, sq.cuda, dtype)

        def test_device_pref(self) :
            an = self.new_annealer(10, 10)
            prefs = an.get_preferences()
            self.assertEqual(prefs['device'], 'cuda')

        def test_set_algorithm(self) :
            ann = self.new_annealer(10, 10)
            ann.set_preferences(algorithm = sq.algorithm.default)
            pref = ann.get_preferences()
            self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

            ann.set_preferences(algorithm = sq.algorithm.naive)
            pref = ann.get_preferences()
            self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

            ann.set_preferences(algorithm = sq.algorithm.coloring)
            pref = ann.get_preferences()
            self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)


    class TestCUDADenseGraphAnnealerFP32(TestCUDADenseGraphAnnealerBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDADenseGraphAnnealerBase.__init__(self, np.float32)
            unittest.TestCase.__init__(self, testFunc)
            

    class TestCUDADenseGraphAnnealerFP64(TestCUDADenseGraphAnnealerBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDADenseGraphAnnealerBase.__init__(self, np.float64)
            unittest.TestCase.__init__(self, testFunc)

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
