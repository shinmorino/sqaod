import unittest
import numpy as np
import sqaod as sq
from example_problems import *

class TestCommon(unittest.TestCase):

    def run_annealer(self, ann) :
        ann.randomize_q()
        sq.anneal(ann)
        ann.randomize_q()
        sq.anneal(ann)

    def run_searcher(self, solver) :
        solver.search()

    def run_dense_graph_searcher(self, solver, dtype) :
        solver.set_problem(W, sq.minimize)
        solver.search()
        solver.set_problem(W, sq.maximize)
        solver.search()

    def test_generate_dense_graph_problem(self):
        W = dense_graph_random(8, dtype=np.float64)
        self.assertTrue(sq.is_symmetric(W))

    def test_generate_bipartite_graph_problem(self):
        b0, b1, W = bipartite_graph_random(8, 8, dtype=np.float64)
        
    def test_dense_graph_solvers(self):
        W = dense_graph_random(8, dtype=np.float64)
        
        ann = sq.py.dense_graph_annealer(W, sq.minimize, 4)
        self.run_annealer(ann)
        ann = sq.py.dense_graph_annealer(W, sq.maximize, 4)
        self.run_annealer(ann)

        ann = sq.py.dense_graph_bf_solver(W, sq.minimize)
        self.run_searcher(ann)
        ann = sq.py.dense_graph_bf_solver(W, sq.maximize)
        self.run_searcher(ann)
        
        ann = sq.cpu.dense_graph_annealer(W, 4, sq.minimize, np.float64)
        self.run_annealer(ann)
        ann = sq.cpu.dense_graph_annealer(W, 4, sq.maximize, np.float64)
        self.run_annealer(ann)
        
        ann = sq.cpu.dense_graph_bf_solver(W, sq.minimize, np.float64)
        self.run_searcher(ann)
        ann = sq.cpu.dense_graph_bf_solver(W, sq.maximize, np.float64)
        self.run_searcher(ann)

    def test_bipartite_graph_solvers(self):
        b0, b1, W = bipartite_graph_random(2, 2, np.float64)
        
        ann = sq.py.bipartite_graph_annealer(b0, b1, W, sq.minimize, 2)
        self.run_annealer(ann)
        ann = sq.py.bipartite_graph_annealer(b0, b1, W, sq.maximize, 2)
        self.run_annealer(ann)

        ann = sq.py.bipartite_graph_bf_solver(b0, b1, W, sq.minimize)
        self.run_searcher(ann)
        ann = sq.py.bipartite_graph_bf_solver(b0, b1, W, sq.maximize)
        self.run_searcher(ann)
        
        ann = sq.cpu.bipartite_graph_annealer(b0, b1, W, sq.minimize, 2, np.float64)
        self.run_annealer(ann)
        ann = sq.cpu.bipartite_graph_annealer(b0, b1, W, sq.maximize, 2, np.float64)
        self.run_annealer(ann)

        ann = sq.cpu.bipartite_graph_bf_solver(b0, b1, W, sq.minimize, np.float64)
        self.run_searcher(ann)
        ann = sq.cpu.bipartite_graph_bf_solver(b0, b1, W, sq.maximize, np.float64)
        self.run_searcher(ann)
        
if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
