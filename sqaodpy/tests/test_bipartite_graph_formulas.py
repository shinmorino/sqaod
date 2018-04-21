from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from .example_problems import *


class TestBipartiteGraphFormulasBase :

    def __init__(self, pkg, dtype) :
        self.pkg = pkg
        self.dtype = dtype
        self.epu = 1.e-5 if dtype == np.float32 else 1.e-12

    def new_QUBO(self, N0, N1) :
        return bipartite_graph_random(N0, N1, dtype=np.float64)

    def create_sequence(self, N0, N1) :
        N = N0 + N1
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        x0, x1 = np.empty((2 ** N, N0), np.int8), np.empty((2 ** N, N1), np.int8)
        for idx in range(2 ** N) :
            x0[idx][...] = xlist[idx][0:N0]
            x1[idx][...] = xlist[idx][N0:]
        return x0, x1
    
    def test_engery_with_zero_x(self):
        N0, N1 = 8, 8
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = np.zeros(N0, np.int8), np.zeros(N1, np.int8)
        Equbo = self.pkg.formulas.bipartite_graph_calculate_E(b0, b1, W, x0, x1, self.dtype)
        self.assertEqual(Equbo, 0.)

    def test_engery_with_one_x(self):
        N0, N1 = 8, 8
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = np.ones(N0, np.int8), np.ones(N1, np.int8)
        Equbo = self.pkg.formulas.bipartite_graph_calculate_E(b0, b1, W, x0, x1, self.dtype)
        self.assertTrue(np.allclose(Equbo, np.sum(b0) + np.sum(b1) + np.sum(W)))

    def test_engery(self):
        N0, N1 = 4, 8
        N = N0 + N1
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = self.create_sequence(N0, N1)
        
        Eref = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                                   x0, x1, self.dtype)
        E = np.empty((2 ** N))
        for i in range(0, 2 ** N) :
            E[i] = self.pkg.formulas.bipartite_graph_calculate_E(b0, b1, W,
                                                                 x0[i], x1[i], self.dtype)
        self.assertTrue(np.allclose(Eref, E, atol=self.epu))

    def test_engery_batch_1(self):
        N0, N1 = 4, 8
        N = N0 + N1
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = self.create_sequence(N0, N1)
        
        Eref = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                                   x0, x1, self.dtype)
        E = np.empty((2 ** N))
        for i in range(0, 2 ** N) :
            E[i] = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                            x0[i], x1[i], self.dtype)[0]
        self.assertTrue(np.allclose(Eref, E, atol=self.epu))
        
    def test_engery_with_x_batched(self):
        N0, N1 = 4, 8
        N = N0 + N1
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = self.create_sequence(N0, N1)
        
        Eref = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                                   x0, x1, self.dtype)
        E = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, self.dtype)
        self.assertTrue(np.allclose(Eref, E))
        
    def test_hamiltonian_energy(self):
        N0, N1 = 4, 8
        N = N0 + N1
        b0, b1, W = self.new_QUBO(N0, N1)
        h0, h1, J, c = \
               self.pkg.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h0) - np.sum(h1) + np.sum(J) + c, atol=self.epu))

        x0, x1 = self.create_sequence(N0, N1)
        q0, q1 = x0 * 2 - 1, x1 * 2 - 1
        
        Eref = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                                   x0, x1, self.dtype)
        Eising = np.empty((2 ** N))
        for idx in range(0, 2 ** N) :
            Eising[idx] = self.pkg.formulas.bipartite_graph_calculate_E_from_spin(h0, h1, J, c,
                                                                    q0[idx], q1[idx], self.dtype)
        self.assertTrue(np.allclose(Eref, Eising, atol=self.epu))
        
    def test_hamiltonian_energy_batch_1(self):
        N0, N1 = 4, 8
        N = N0 + N1
        b0, b1, W = self.new_QUBO(N0, N1)
        h0, h1, J, c = \
               self.pkg.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h0) - np.sum(h1) + np.sum(J) + c, atol=self.epu))

        x0, x1 = self.create_sequence(N0, N1)
        q0, q1 = x0 * 2 - 1, x1 * 2 - 1
        
        Eref = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                                   x0, x1, self.dtype)
        Eising = np.empty((2 ** N))
        for idx in range(0, 2 ** N) :
            Eising[idx] = \
                self.pkg.formulas.bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c,
                                                                q0[idx], q1[idx], self.dtype)[0]
        self.assertTrue(np.allclose(Eref, Eising, atol=self.epu))
        
    def test_hamiltonian_energy_batched(self):
        N0, N1 = 4, 8
        N = N0 + N1
        b0, b1, W = self.new_QUBO(N0, N1)
        h0, h1, J, c = \
               self.pkg.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h0) - np.sum(h1) + np.sum(J) + c, atol=self.epu))

        x0, x1 = self.create_sequence(N0, N1)
        q0, q1 = x0 * 2 - 1, x1 * 2 - 1
        
        Eref = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                                   x0, x1, self.dtype)
        Eising = self.pkg.formulas.bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c,
                                                                        q0, q1, self.dtype)
        self.assertTrue(np.allclose(Eref, Eising, atol=self.epu))
        

# Tests for py.formulas
class TestPyBipartiteGraphFormulas(TestBipartiteGraphFormulasBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestBipartiteGraphFormulasBase.__init__(self, sq.py, np.float64)
        unittest.TestCase.__init__(self, testFunc)


# Tests for native formula modules
class TestNativeBipartiteGraphFormulasBase(TestBipartiteGraphFormulasBase) :
    def __init__(self, pkg, dtype) :
        TestBipartiteGraphFormulasBase.__init__(self, pkg, dtype)

    def test_compare_engery(self):
        N0, N1 = 4, 8
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = self.create_sequence(N0, N1)
        q0, q1 = x0 * 2 - 1, x1 * 2 - 1
        
        Eref = sq.py.formulas.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, self.dtype)

        N = N0 + N1
        E = np.empty((2 ** N))
        for i in range(0, 2 ** N) :
            E[i] = self.pkg.formulas.bipartite_graph_calculate_E(b0, b1, W,
                                                        x0[i], x1[i], self.dtype)
        self.assertTrue(np.allclose(Eref, E, atol=self.epu))

    def test_compare_engery_batched(self):
        N0, N1 = 4, 8
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = self.create_sequence(N0, N1)
        q0, q1 = x0 * 2 - 1, x1 * 2 - 1
        
        Eref = sq.py.formulas.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, self.dtype)
        E = self.pkg.formulas.bipartite_graph_batch_calculate_E(b0, b1, W,
                                                                x0, x1, self.dtype)
        self.assertTrue(np.allclose(Eref, E, atol=self.epu))        

    def test_compare_hamiltonian(self):
        N0, N1 = 4, 8
        b0, b1, W = self.new_QUBO(N0, N1)
        h00, h01, J0, c0 = \
            sq.py.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W, self.dtype)
        h10, h11, J1, c1 = \
            self.pkg.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h10)- np.sum(h11) + np.sum(J1) + c1, atol=self.epu))
        self.assertTrue(np.allclose(h00, h10))
        self.assertTrue(np.allclose(h01, h11))
        self.assertTrue(np.allclose(J0, J1))
        self.assertTrue(np.allclose(c0, c1))

    def test_compare_hamiltonian_energy(self):
        N0, N1 = 4, 8
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = self.create_sequence(N0, N1)
        q0, q1 = x0 * 2 - 1, x1 * 2 - 1
        
        Eref = sq.py.formulas.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, self.dtype)
        
        h0, h1, J, c = \
            self.pkg.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W, self.dtype)
        N = N0 + N1
        E = np.empty((2 ** N))
        for i in range(0, 2 ** N) :
            E[i] = \
                self.pkg.formulas.bipartite_graph_calculate_E_from_spin(h0, h1, J, c,
                                                                q0[i], q1[i], self.dtype)
        self.assertTrue(np.allclose(Eref, E, atol=self.epu))

    def test_compare_hamiltonian_energy_batched(self):
        N0, N1 = 4, 8
        b0, b1, W = self.new_QUBO(N0, N1)
        x0, x1 = self.create_sequence(N0, N1)
        q0, q1 = x0 * 2 - 1, x1 * 2 - 1
        
        Eref = sq.py.formulas.bipartite_graph_batch_calculate_E(b0, b1, W, x0, x1, self.dtype)
        
        h0, h1, J, c = \
            self.pkg.formulas.bipartite_graph_calculate_hamiltonian(b0, b1, W, self.dtype)
        N = N0 + N1
        E = self.pkg.formulas.bipartite_graph_batch_calculate_E_from_spin(h0, h1, J, c,
                                                                    q0, q1, self.dtype)
        self.assertTrue(np.allclose(Eref, E, atol=self.epu))


class TestCPUBipartiteGraphFormulasFP32(TestNativeBipartiteGraphFormulasBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestNativeBipartiteGraphFormulasBase.__init__(self, sq.cpu, np.float32)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUBipartiteGraphFormulasFP64(TestBipartiteGraphFormulasBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestBipartiteGraphFormulasBase.__init__(self, sq.cpu, np.float64)
        unittest.TestCase.__init__(self, testFunc)


if False: #sq.is_cuda_available() :

    class TestCUDABipartiteGraphFormulasFP32(TestBipartiteGraphFormulasBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestBipartiteGraphFormulasBase.__init__(self, sq.cuda, np.float32)
            unittest.TestCase.__init__(self, testFunc)

    class TestCUDABipartiteGraphFormulasFP64(TestBipartiteGraphFormulasBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestBipartiteGraphFormulasBase.__init__(self, sq.cuda, np.float64)
            unittest.TestCase.__init__(self, testFunc)
            

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
