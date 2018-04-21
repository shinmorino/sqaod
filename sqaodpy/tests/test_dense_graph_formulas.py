from __future__ import print_function
import unittest
import numpy as np
import sqaod as sq
import sqaod.common as common
from example_problems import *


class TestDenseGraphFormulasBase :

    def __init__(self, pkg, dtype) :
        self.pkg = pkg
        self.dtype = dtype
        self.epu = 4.e-5 if dtype == np.float32 else 1.e-8

    def new_W(self, N) :
        return common.generate_random_symmetric_W((N), dtype=np.float64)
    
    def test_engery_with_zero_x(self):
        N = 8
        W = self.new_W(N)
        x = np.zeros(N, np.int8)
        Equbo = self.pkg.formulas.dense_graph_calculate_E(W, x, self.dtype)
        self.assertEqual(Equbo, 0.)

    def test_engery_with_one_x(self):
        N = 8
        W = self.new_W(N)
        x = np.ones(N, np.int8)
        Equbo = self.pkg.formulas.dense_graph_calculate_E(W, x, self.dtype)
        self.assertTrue(np.allclose(Equbo, np.sum(W)))

    def test_engery(self):
        N = 8
        W = self.new_W(N)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        Ebatched = self.pkg.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        E = np.ones((2 ** N))
        for i in range(0, 2 ** N) :
            E[i] = self.pkg.formulas.dense_graph_calculate_E(W, xlist[i], self.dtype)
        self.assertTrue(np.allclose(Ebatched, E, atol=self.epu))

    def test_energy_batch_1(self):
        N = 8
        W = self.new_W(N)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        Ebatched = self.pkg.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        E = np.empty((2 ** N))
        for i in range(0, 2 ** N) :
            E[i] = self.pkg.formulas.dense_graph_batch_calculate_E(W, xlist[i], self.dtype)[0]
        self.assertTrue(np.allclose(Ebatched, E, atol=self.epu))

    def test_hamiltonian_energy(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        h, J, c = self.pkg.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h) + np.sum(J) + c, atol=self.epu))
        
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        for idx in range(0, 2 ** N) :
            x = xlist[idx]
            Equbo = self.pkg.formulas.dense_graph_calculate_E(W, x, self.dtype)
            q = 2 * x - 1
            Eising = self.pkg.formulas.dense_graph_calculate_E_from_spin(h, J, c, q, self.dtype)
            self.assertTrue(np.allclose(Equbo, Eising, atol=self.epu))

    def test_hamiltonian_energy_batched(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        h, J, c = self.pkg.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h) + np.sum(J) + c, atol=self.epu))

        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        Equbo = self.pkg.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        qlist = 2 * xlist - 1
        Eising = sq.py.formulas.dense_graph_batch_calculate_E_from_spin(h, J, c, qlist, self.dtype)
        self.assertTrue(np.allclose(Equbo, Eising, atol=self.epu))

        
# Tests for py.formulas
class TestPyDenseGraphFormulas(TestDenseGraphFormulasBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestDenseGraphFormulasBase.__init__(self, sq.py, np.float64)
        unittest.TestCase.__init__(self, testFunc)


# Tests for native formula modules
class TestNativeDenseGraphFormulasBase(TestDenseGraphFormulasBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestDenseGraphFormulasBase.__init__(self, sq.py, np.float64)
        unittest.TestCase.__init__(self, testFunc)

    def test_compare_engery(self):
        N = 8
        W = self.new_W(N)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        Ebatched = sq.py.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        E = np.empty((2 ** N))
        for i in range(0, 2 ** N) :
            E[i] = self.pkg.formulas.dense_graph_calculate_E(W, xlist[i], self.dtype)
        self.assertTrue(np.allclose(Ebatched, E))

    def test_compare_engery_batched(self):
        N = 8
        W = self.new_W(N)
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        Ebatched = sq.py.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        E = self.pkg.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        self.assertTrue(np.allclose(Ebatched, E))

    def test_compare_hamiltonian(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        h0, J0, c0 = sq.py.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)
        h1, J1, c1 = self.pkg.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h1) + np.sum(J1) + c1, atol=self.epu))
        self.assertTrue(np.allclose(h0, h1))
        self.assertTrue(np.allclose(J0, J1))
        self.assertTrue(np.allclose(c0, c1))

    def test_compare_hamiltonian_energy(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        h0, J0, c0 = sq.py.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)
        h1, J1, c1 = self.pkg.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h1) + np.sum(J1) + c1, atol=self.epu))
        
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        Equbo = sq.py.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        Eising = np.empty((2 ** N))
        for idx in range(0, 2 ** N) :
            x = xlist[idx]
            q = 2 * x - 1
            Eising[idx] = self.pkg.formulas.dense_graph_calculate_E_from_spin(h1, J1, c1, q, self.dtype)
        self.assertTrue(np.allclose(Equbo, Eising, atol=self.epu))

    def test_compare_hamiltonian_energy_batched(self):
        N = 8
        W = common.generate_random_symmetric_W((N), dtype=np.float64)
        h0, J0, c0 = sq.py.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)
        h1, J1, c1 = self.pkg.formulas.dense_graph_calculate_hamiltonian(W, self.dtype)

        # identity
        self.assertTrue(np.allclose(0., - np.sum(h1) + np.sum(J1) + c1, atol=self.epu))
        
        xlist = common.create_bitset_sequence(range(0, 2 ** N), N)
        Equbo = sq.py.formulas.dense_graph_batch_calculate_E(W, xlist, self.dtype)
        Eising = np.empty((2 ** N))
        qlist = 2 * xlist - 1
        Eising = self.pkg.formulas.dense_graph_batch_calculate_E_from_spin(h1, J1, c1, qlist, self.dtype)
        self.assertTrue(np.allclose(Equbo, Eising, atol=self.epu))
    

class TestCPUDenseGraphFormulasFP32(TestDenseGraphFormulasBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestDenseGraphFormulasBase.__init__(self, sq.cpu, np.float32)
        unittest.TestCase.__init__(self, testFunc)

class TestCPUDenseGraphFormulasFP64(TestDenseGraphFormulasBase, unittest.TestCase) :
    def __init__(self, testFunc) :
        TestDenseGraphFormulasBase.__init__(self, sq.cpu, np.float64)
        unittest.TestCase.__init__(self, testFunc)


if sq.is_cuda_available() :

    class TestCUDADenseGraphFormulasFP32(TestDenseGraphFormulasBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestDenseGraphFormulasBase.__init__(self, sq.cuda, np.float32)
            unittest.TestCase.__init__(self, testFunc)

    class TestCUDADenseGraphFormulasFP64(TestDenseGraphFormulasBase, unittest.TestCase) :
        def __init__(self, testFunc) :
            TestDenseGraphFormulasBase.__init__(self, sq.cuda, np.float64)
            unittest.TestCase.__init__(self, testFunc)
            

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
