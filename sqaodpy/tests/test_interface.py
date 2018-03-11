import unittest
import numpy as np
import sqaod as sq
from example_problems import *

class TestCommon(unittest.TestCase):

    def run_annealer(self, ann) :
        sq.anneal(ann)
        sq.anneal(ann)

    def run_searcher(self, searcher) :
        searcher.search()

    def run_dense_graph_searcher(self, searcher, dtype) :
        searcher.set_qubo(W, sq.minimize)
        searcher.search()
        searcher.set_qubo(W, sq.maximize)
        searcher.search()

    def test_generate_dense_graph_problem(self):
        W = dense_graph_random(8, dtype=np.float64)
        self.assertTrue(sq.is_symmetric(W))

    def test_generate_bipartite_graph_problem(self):
        b0, b1, W = bipartite_graph_random(8, 8, dtype=np.float64)
        
    def test_dense_graph_searcher(self):
        W = dense_graph_random(8, dtype=np.float64)
        
        ann = sq.py.dense_graph_annealer(W, sq.minimize, n_trotters=4)
        self.run_annealer(ann)
        ann = sq.py.dense_graph_annealer(W, sq.maximize, n_trotters=4)
        self.run_annealer(ann)

        ann = sq.py.dense_graph_bf_searcher(W, sq.minimize)
        self.run_searcher(ann)
        ann = sq.py.dense_graph_bf_searcher(W, sq.maximize)
        self.run_searcher(ann)
        
        ann = sq.cpu.dense_graph_annealer(W, sq.minimize, np.float64, n_trotters=4)
        self.run_annealer(ann)
        ann = sq.cpu.dense_graph_annealer(W, sq.maximize, np.float64, n_trotters=4)
        self.run_annealer(ann)
        
        ann = sq.cpu.dense_graph_bf_searcher(W, sq.minimize, np.float64)
        self.run_searcher(ann)
        ann = sq.cpu.dense_graph_bf_searcher(W, sq.maximize, np.float64)
        self.run_searcher(ann)

    def test_bipartite_graph_solvers(self):
        b0, b1, W = bipartite_graph_random(2, 2, np.float64)
        
        ann = sq.py.bipartite_graph_annealer(b0, b1, W, sq.minimize, n_trotters=2)
        self.run_annealer(ann)
        ann = sq.py.bipartite_graph_annealer(b0, b1, W, sq.maximize, n_trotters=2)
        self.run_annealer(ann)

        ann = sq.py.bipartite_graph_bf_searcher(b0, b1, W, sq.minimize)
        self.run_searcher(ann)
        ann = sq.py.bipartite_graph_bf_searcher(b0, b1, W, sq.maximize)
        self.run_searcher(ann)
        
        ann = sq.cpu.bipartite_graph_annealer(b0, b1, W, sq.minimize, np.float64, n_trotters=2)
        self.run_annealer(ann)
        ann = sq.cpu.bipartite_graph_annealer(b0, b1, W, sq.maximize, np.float64, n_trotters=2)
        self.run_annealer(ann)

        ann = sq.cpu.bipartite_graph_bf_searcher(b0, b1, W, sq.minimize, np.float64)
        self.run_searcher(ann)
        ann = sq.cpu.bipartite_graph_bf_searcher(b0, b1, W, sq.maximize, np.float64)
        self.run_searcher(ann)
        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
