import numpy as np
import sqaod as sq
from tests import example_problems
import unittest

class TestSymmetrizeBase :
    def __init__(self) :
        assert False, 'Namescope to hide unittest base class'


    class TestDGAnnealer(unittest.TestCase) :

        def test_symmetrize_triu(self) :
            sol = self.pkg.dense_graph_annealer(dtype = self.dtype)
            W = example_problems.dense_graph_random(self.N, self.dtype)
            sol.set_qubo(W)
            h0, J0, c0 = sol.get_hamiltonian()
            
            Wtriu = sq.common.symmetric_to_triu(W)
            sol.set_qubo(Wtriu)
            h1, J1, c1 = sol.get_hamiltonian()

            self.assertTrue(np.allclose(J0, J1))

        def test_symmetrize_tril(self) :
            sol = self.pkg.dense_graph_annealer(dtype = self.dtype)
            W = example_problems.dense_graph_random(self.N, self.dtype)
            sol.set_qubo(W)
            h0, J0, c0 = sol.get_hamiltonian()

            Wtril = sq.common.symmetric_to_tril(W)
            sol.set_qubo(Wtril)
            h1, J1, c1 = sol.get_hamiltonian()

            self.assertTrue(np.allclose(J0, J1))

        def test_symmetrize_triu_hamiltonian(self) :
            sol = self.pkg.dense_graph_annealer(dtype = self.dtype)
            W = example_problems.dense_graph_random(self.N, self.dtype)
            sol.set_qubo(W)
            h0, J0, c0 = self.pkg.formulas.dense_graph_calculate_hamiltonian(W, W.dtype)

            J0u = sq.common.symmetric_to_triu(J0)
            sol.set_hamiltonian(h0, J0u, c0)
            h1, J1, c1 = sol.get_hamiltonian()

            self.assertTrue(np.allclose(J0, J1))

        def test_symmetrize_tril_hamiltonian(self) :
            sol = self.pkg.dense_graph_annealer(dtype = self.dtype)
            W = example_problems.dense_graph_random(self.N, self.dtype)
            sol.set_qubo(W)
            h0, J0, c0 = self.pkg.formulas.dense_graph_calculate_hamiltonian(W, W.dtype)

            J0l = sq.common.symmetric_to_tril(J0)
            sol.set_hamiltonian(h0, J0l, c0)
            h1, J1, c1 = sol.get_hamiltonian()

            self.assertTrue(np.allclose(J0, J1))

        
    class Test_DGBFSearcher(unittest.TestCase) :

        def test_symmetrize_triu(self) :
            sol = self.pkg.dense_graph_bf_searcher(dtype = self.dtype)
            W = example_problems.dense_graph_random(self.N, self.dtype)
            sol.set_qubo(W)
            sol.search()
            x = sol.get_x()
            E = sol.get_E()

            Wu = sq.common.symmetric_to_triu(W)
            sol.set_qubo(Wu)
            sol.search()
            xu = sol.get_x()
            Eu = sol.get_E()

            self.assertTrue(np.allclose(E, Eu))
            self.assertTrue(np.allclose(x, xu))

            Wl = sq.common.symmetric_to_tril(W)
            sol.set_qubo(Wl)
            sol.search()
            xl = sol.get_x()
            El = sol.get_E()

            self.assertTrue(np.allclose(E, El))
            self.assertTrue(np.allclose(x, xl))


class TestSymmetrize_DGAnnealerPy(TestSymmetrizeBase.TestDGAnnealer) :
    N = 512
    dtype = np.float64
    pkg = sq.py

class TestSymmetrize_DGAnnealerCPU64(TestSymmetrizeBase.TestDGAnnealer) :
    N = 512
    dtype = np.float64
    pkg = sq.cpu

class TestSymmetrize_DGAnnealerCPU32(TestSymmetrizeBase.TestDGAnnealer) :
    N = 512
    dtype = np.float32
    pkg = sq.cpu

if sq.is_cuda_available() :
    class TestSymmetrize_DGAnnealerCUDA64(TestSymmetrizeBase.TestDGAnnealer) :
        N = 512
        dtype = np.float64
        pkg = sq.cuda
    
    class TestSymmetrize_DGAnnealerCUDA32(TestSymmetrizeBase.TestDGAnnealer) :
        N = 512
        dtype = np.float32
        pkg = sq.cuda
        

class TestSymmetrize_DGBFSearcherPy(TestSymmetrizeBase.Test_DGBFSearcher) :
    N = 8
    dtype = np.float64
    pkg = sq.py


class TestSymmetrize_DGBFSearcherCPU64(TestSymmetrizeBase.Test_DGBFSearcher) :
    N = 16
    dtype = np.float64
    pkg = sq.cpu

class TestSymmetrize_DGBFSearcherCPU32(TestSymmetrizeBase.Test_DGBFSearcher) :
    N = 16
    dtype = np.float32
    pkg = sq.cpu
    
if sq.is_cuda_available() :
    class TestSymmetrize_DGBFSearcherCUDA64(TestSymmetrizeBase.Test_DGBFSearcher) :
        N = 16
        dtype = np.float64
        pkg = sq.cuda
    
    class TestSymmetrize_DGBFSearcherCUDA32(TestSymmetrizeBase.Test_DGBFSearcher) :
        N = 16
        dtype = np.float32
        pkg = sq.cuda
        

if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
