from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from .example_problems import *
from math import log, exp

class TestBipartiteGraphAnnealerBase:

    def __init__(self, anpkg, dtype) :
        self.anpkg = anpkg
        self.dtype = dtype
        self.epu = 5.e-5 if dtype == np.float32 else 1.e-8

    def new_annealer(self, N0, N1, m) :
        an = self.anpkg.bipartite_graph_annealer(dtype=self.dtype)
        b0, b1, W = bipartite_graph_random(N0, N1, self.dtype)
        an.set_qubo(b0, b1, W)
        an.set_preferences(n_trotters = m)
        return an

    def test_calling_sequence(self) :
        N0, N1 = 200, 100
        m = 100
        an = self.new_annealer(N0, N1, m)
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
        xpair = an.get_x()
        h0, h1, J, c = an.get_hamiltonian()
        qpair = an.get_q()

    def test_problem_size(self) :
        N0, N1 = 100, 110
        m = 10
        an = self.new_annealer(N0, N1, m)
        N0out, N1out = an.get_problem_size()
        self.assertEqual(N0, N0out)
        self.assertEqual(N1, N1out)

    def test_set_n_trotters(self) :
        an = self.new_annealer(10, 10, 10)
        an.set_preferences(n_trotters = 3)
        prefs = an.get_preferences()
        self.assertEqual(prefs['n_trotters'], 3)
        
    def test_get_hamiltonian(self) :
        N0, N1 = 10, 11
        m = 10
        an = self.anpkg.bipartite_graph_annealer(dtype=self.dtype)
        b0, b1, W = bipartite_graph_random(N0, N1, self.dtype)
        an.set_qubo(b0, b1, W)
        h00, h01, J0, c0 = an.get_hamiltonian()
        h10, h11, J1, c1 = sq.py.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W)
        # print(h00, h10)
        self.assertTrue(np.allclose(h00, h10, atol=self.epu))
        self.assertTrue(np.allclose(h01, h11))
        self.assertTrue(np.allclose(J0, J1))
        self.assertTrue(np.allclose(c0, c1)) #, atol=self.epu))

    def test_set_hamiltonian(self) :
        N0, N1 = 10, 11
        m = 10
        b0, b1, W = bipartite_graph_random(N0, N1, self.dtype)
        h00, h01, J0, c0 = sq.py.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W)
        
        an = self.anpkg.bipartite_graph_annealer(dtype=self.dtype)
        an.set_hamiltonian(h00, h01, J0, c0)
        an.prepare()
        
        h10, h11, J1, c1 = an.get_hamiltonian()
        #print(h00, h10)
        self.assertTrue(np.allclose(h00, h10, atol=self.epu))
        self.assertTrue(np.allclose(h01, h11))
        self.assertTrue(np.allclose(J0, J1))
        self.assertTrue(np.allclose(c0, c1)) #, atol=self.epu))
        
    def test_min_energy(self):
        N0, N1 = 200, 350
        m = 1
        an = self.new_annealer(N0, N1, m)
        an.prepare()

        q0 = np.ndarray((N0), np.int8)
        q1 = np.ndarray((N1), np.int8)
        q0[:] = -1
        q1[:] = -1
        an.set_q((q0, q1))
        an.calculate_E()
        E = an.get_E()
        res = np.allclose(E, 0., atol=self.epu)
        if not res :
            print(E)
        self.assertTrue(res)
        # print(an.E)

    def test_qubo_energy(self):
        N0, N1 = 8, 5
        m = 1
        
        an = self.anpkg.bipartite_graph_annealer(dtype=self.dtype)
        b0, b1, W = bipartite_graph_random(N0, N1, self.dtype)
        an.set_qubo(b0, b1, W)
        an.set_preferences(n_trotters=m)
        an.prepare()

        iMax = 1 << N0
        jMax = 1 << N1

        for i in range(iMax) :
            x0 = common.create_bitset_sequence((i,), N0)
            q0 = sq.bit_to_spin(x0)
            for j in range(jMax) :
                x1 = common.create_bitset_sequence((j,), N1)
                q1 = sq.bit_to_spin(x1)
                Ebf = np.dot(b0, x0.transpose()) + np.dot(b1, x1.transpose()) \
                      + np.dot(x1, np.matmul(W, x0.transpose()))
                an.set_q((q0, q1))
                an.calculate_E()
                Ean = an.get_E()
                if not np.allclose(Ebf, Ean, atol=self.epu) :
                    print(i, j, Ebf, Ean)
                self.assertTrue(np.allclose(Ebf,  Ean, atol=self.epu))
        
    def test_set_q(self):
        N0, N1 = 100, 50
        m = 150
        an = self.new_annealer(N0, N1, m)
        an.prepare()
        q0in = 2 * np.random.randint(0, 2, N0) - 1
        q1in = 2 * np.random.randint(0, 2, N1) - 1
        an.set_q((q0in, q1in))
        qout = an.get_q()

        res = True
        for qpair in qout :
            res &= np.allclose(q0in, qpair[0]) and np.allclose(q1in, qpair[1])
        self.assertTrue(res)
            
    def test_set_q_m(self):
        N0, N1 = 100, 50
        m = 150
        an = self.new_annealer(N0, N1, m)
        an.prepare()
        an.set_preferences(n_trotters = m)

        qin = []
        for loop in range(0, m) :
            q0 = 2 * np.random.randint(0, 2, N0) - 1
            q1 = 2 * np.random.randint(0, 2, N1) - 1
            qin.append((q0.astype(np.int8), q1.astype(np.int8)))
        an.set_q(qin)
        qout = an.get_q()

        self.assertTrue(len(qin) == len(qout))
        res = True
        for qinpair, qoutpair in zip(qin, qout) :
            res &= np.allclose(qinpair[0], qoutpair[0]) and np.allclose(qinpair[1], qoutpair[1])
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
        N0, N1 = 10, 8
        m = 6
        an = self.anpkg.bipartite_graph_annealer(dtype=self.dtype)
        b0 = np.ones((N0), dtype=self.dtype)
        b1 = np.ones((N1), dtype=self.dtype)
        W = np.ones((N1, N0), dtype=self.dtype)
        an.set_qubo(b0, b1, W, sq.minimize)
        an.set_preferences(n_trotters = m)
        an.set_preferences(algorithm = algorithm)

        self.anneal(an)
        
        E = an.get_E()
        self.assertEqual(E.min(), 0)

    def _test_anneal_hamiltonian(self, algorithm) :
        N0, N1 = 10, 8
        m = 6
        an = self.anpkg.bipartite_graph_annealer(dtype=self.dtype)
        b0 = np.ones((N0), dtype=self.dtype)
        b1 = np.ones((N1), dtype=self.dtype)
        W = np.ones((N1, N0), dtype=self.dtype)
        h0, h1, J, c = sq.py.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W)
        an.set_hamiltonian(h0, h1, J, c)
        an.set_preferences(n_trotters = m)
        an.set_preferences(algorithm = algorithm)

        self.anneal(an)

        E = an.get_E()
        self.assertEqual(E.min(), 0)

    def _test_anneal_maximize(self, algorithm) :
        N0, N1 = 10, 8
        m = 6
        an = self.anpkg.bipartite_graph_annealer(dtype=self.dtype)
        b0 = - np.ones((N0), dtype=self.dtype)
        b1 = - np.ones((N1), dtype=self.dtype)
        W = - np.ones((N1, N0), dtype=self.dtype)
        an.set_qubo(b0, b1, W, sq.maximize)
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

        

class TestNativeBipartiteGraphAnnealerBase(TestBipartiteGraphAnnealerBase) :
    def __init__(self, anpkg, dtype) :
        TestBipartiteGraphAnnealerBase.__init__(self, anpkg, dtype)

    def test_precision(self) :
        an = self.new_annealer(10, 10, 1)
        self.assertEqual(an.dtype, self.dtype)

class TestCPUBipartiteGraphAnnealerBase(TestNativeBipartiteGraphAnnealerBase) :
    def __init__(self, dtype) :
        TestNativeBipartiteGraphAnnealerBase.__init__(self, sq.cpu, dtype)

    def test_device_pref(self) :
        an = self.new_annealer(10, 10, 1)
        prefs = an.get_preferences()
        self.assertEqual(prefs['device'], 'cpu')

    def test_set_algorithm(self) :
        ann = self.new_annealer(10, 10, 1)
        ann.set_preferences(algorithm = sq.algorithm.default)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

        ann.set_preferences(algorithm = sq.algorithm.naive)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.naive)

        ann.set_preferences(algorithm = sq.algorithm.coloring)
        pref = ann.get_preferences()
        self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)


        
class TestPyBipartiteGraphAnnealer(TestBipartiteGraphAnnealerBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestBipartiteGraphAnnealerBase.__init__(self, sq.py, np.float64)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUBipartiteGraphAnnealerFP32(TestCPUBipartiteGraphAnnealerBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUBipartiteGraphAnnealerBase.__init__(self, np.float32)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUBipartiteGraphAnnealerFP64(TestCPUBipartiteGraphAnnealerBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestCPUBipartiteGraphAnnealerBase.__init__(self, np.float64)
        unittest.TestCase.__init__(self, testFunc)

if sq.is_cuda_available() :

    class TestCUDABipartiteGraphAnnealerBase(TestNativeBipartiteGraphAnnealerBase) :
        def __init__(self, dtype) :
            TestNativeBipartiteGraphAnnealerBase.__init__(self, sq.cuda, dtype)

        def test_device_pref(self) :
            an = self.new_annealer(10, 10, 1)
            prefs = an.get_preferences()
            self.assertEqual(prefs['device'], 'cuda')

        def test_set_algorithm(self) :
            ann = self.new_annealer(10, 10, 1)
            ann.set_preferences(algorithm = sq.algorithm.default)
            pref = ann.get_preferences()
            self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

            ann.set_preferences(algorithm = sq.algorithm.naive)
            pref = ann.get_preferences()
            self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

            ann.set_preferences(algorithm = sq.algorithm.coloring)
            pref = ann.get_preferences()
            self.assertTrue(pref['algorithm'] == sq.algorithm.coloring)

    class TestCUDABipartiteGraphAnnealerFP32(TestCUDABipartiteGraphAnnealerBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDABipartiteGraphAnnealerBase.__init__(self, np.float32)
            unittest.TestCase.__init__(self, testFunc)

    class TestCUDABipartiteGraphAnnealerFP64(TestCUDABipartiteGraphAnnealerBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestCUDABipartiteGraphAnnealerBase.__init__(self, np.float64)
            unittest.TestCase.__init__(self, testFunc)

        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
