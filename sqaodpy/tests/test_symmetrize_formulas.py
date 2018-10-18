import numpy as np
import sqaod as sq
from tests import example_problems
import unittest

class Base :
    def __init__(self) :
        assert False, 'Namescope to hide unittest base class'
        
    class TestDGFormulas(unittest.TestCase) :

        def test_calculate_hamiltonian(self) :
            Wsym = example_problems.dense_graph_random(self.N, self.dtype)
            hsym, Jsym, csym = self.pkg.formulas.dense_graph_calculate_hamiltonian(Wsym, Wsym.dtype)

            Wu = np.triu(Wsym)
            hu, Ju, cu = self.pkg.formulas.dense_graph_calculate_hamiltonian(Wu, Wu.dtype)

            self.assertTrue(np.allclose(hsym, hu))
            self.assertTrue(np.allclose(Jsym, Ju))
            self.assertTrue(np.allclose(csym, cu))

            Wl = np.tril(Wsym)
            hl, Jl, cl = self.pkg.formulas.dense_graph_calculate_hamiltonian(Wl, Wl.dtype)

            self.assertTrue(np.allclose(hsym, hl))
            self.assertTrue(np.allclose(Jsym, Jl))
            self.assertTrue(np.allclose(csym, cl))


        def test_calculate_E(self) :
            Wsym = example_problems.dense_graph_random(self.N, self.dtype)
            x = np.empty((self.N, ), dtype=np.int8)
            sq.randomize_spin(x)
            x = sq.bit_from_spin(x)

            Esym = self.pkg.formulas.dense_graph_calculate_E(Wsym, x, Wsym.dtype)

            Wu = np.triu(Wsym)
            Eu = self.pkg.formulas.dense_graph_calculate_E(Wu, x, Wu.dtype)
            self.assertTrue(np.allclose(Esym, Eu))

            Wl = np.tril(Wsym)
            El = self.pkg.formulas.dense_graph_calculate_E(Wl, x, Wl.dtype)
            self.assertTrue(np.allclose(Esym, El))


        def test_batch_calculate_E(self) :
            Wsym = example_problems.dense_graph_random(self.N, self.dtype)
            x = np.empty((self.N, self.N), dtype=np.int8)
            sq.randomize_spin(x)
            x = sq.bit_from_spin(x)

            Esym = self.pkg.formulas.dense_graph_batch_calculate_E(Wsym, x, Wsym.dtype)

            Wu = np.triu(Wsym)
            Eu = self.pkg.formulas.dense_graph_batch_calculate_E(Wu, x, Wu.dtype)
            self.assertTrue(np.allclose(Esym, Eu))

            Wl = np.tril(Wsym)
            El = self.pkg.formulas.dense_graph_batch_calculate_E(Wl, x, Wl.dtype)
            self.assertTrue(np.allclose(Esym, El))

        def test_calculate_E_from_spin(self) :
            Wsym = example_problems.dense_graph_random(self.N, self.dtype)

            h, Jsym, c = self.pkg.formulas.dense_graph_calculate_hamiltonian(Wsym, Wsym.dtype)
            q = np.empty((self.N, ), dtype=np.int8)
            sq.randomize_spin(q)

            Esym = self.pkg.formulas.dense_graph_calculate_E_from_spin(h, Jsym, c, q, Jsym.dtype)

            Ju = np.triu(Jsym)
            Eu = self.pkg.formulas.dense_graph_calculate_E_from_spin(h, Ju, c, q, Ju.dtype)
            self.assertTrue(np.allclose(Esym, Eu))

            Jl = np.tril(Jsym)
            El = self.pkg.formulas.dense_graph_calculate_E_from_spin(h, Jl, c, q, Jl.dtype)
            self.assertTrue(np.allclose(Esym, El))


        def test_batch_calculate_E_from_spin(self) :
            Wsym = example_problems.dense_graph_random(self.N, self.dtype)

            h, Jsym, c = self.pkg.formulas.dense_graph_calculate_hamiltonian(Wsym, Wsym.dtype)
            q = np.empty((self.N, self.N), dtype=np.int8)
            sq.randomize_spin(q)

            Esym = self.pkg.formulas.dense_graph_batch_calculate_E_from_spin(h, Jsym, c, q, Jsym.dtype)

            Ju = np.triu(Jsym)
            Eu = self.pkg.formulas.dense_graph_batch_calculate_E_from_spin(h, Ju, c, q, Ju.dtype)
            self.assertTrue(np.allclose(Esym, Eu))

            Jl = np.tril(Jsym)
            El = self.pkg.formulas.dense_graph_batch_calculate_E_from_spin(h, Jl, c, q, Jl.dtype)
            self.assertTrue(np.allclose(Esym, El))


class TestDGFormulasPy(Base.TestDGFormulas) :
    N = 512
    dtype = np.float64
    pkg = sq.py

class TestDGFormulasCPU64(Base.TestDGFormulas) :
    N = 512
    dtype = np.float64
    pkg = sq.cpu

class TestDGFormulasCPU32(Base.TestDGFormulas) :
    N = 512
    dtype = np.float32
    pkg = sq.cpu

if sq.is_cuda_available() :
    class TestDGFormulasCUDA64(Base.TestDGFormulas) :
        N = 512
        dtype = np.float64
        pkg = sq.cuda
    
    class TestDGFormulasCUDA32(Base.TestDGFormulas) :
        N = 512
        dtype = np.float32
        pkg = sq.cuda
    
        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
