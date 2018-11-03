Bipartite Graph Brute-force Searcher
====================================

Sqaod implements 3 bipartite graph brute-force searcher classes.
 * `sqaod.py.BipartiteGraphBFSearcher`_
   
   Python-based reference implementation of bipartite graph brute-force search.
   
 * `sqaod.cpu.BipartiteGraphBFSearcher`_
   
   Parallelized CPU implementation by using OpenMP.
   
 * `sqoad.cuda.BipartiteGraphBFSearcher`_
   
   Parallelized and accelearted by using NVIDIA GPU, CUDA.

Factory function
----------------

Every solver class has its own factory function in the same package.
For example, sqaod.py.BipartiteGraphAnnealer is instantiated by `sqaod.py.bipartite_graph_bf_searcher`_.

Method calling sequence
-----------------------

Below is a psudo-code to run bipartite graph brute-force searcher.

.. code-block:: python

   import sqaod as sq

   # Assumption
   # N0, N1 : problem size.
   # dtype : np.float32. Numerical precision is assumed to be np.float32.
   
   # 1. Preparing QUBO matrix, W.
   #  QUBO energy is defined as E = b0 * x0.T + b1 * x1.T + x1 * W * x0.T
   b0 = ... # vector(length = N0)
   b1 = ... # vector(length = N1)
   W = ...  # matrix, shape == (N1, N0)

   # 2. instantiate brute-force searcher
   searcher = sqaod.py.bipartite_graph_bf_searcher(W, sq.minimize, dtype)

   # Alternatively, you're able to set QUBO as shown below.
   #
   # searcher = sqaod.py.bipartite_graph_bf_searcher(dtype)
   # searcher.set_qubo(b0, b1, W, sq.minimize)

   # 3. (optional)
   #    Set preferences. You're able to select tile size.
   #    Preferences are given as **kwargs.
   #
   # searcher.set_preference(tile_size_0 = 1024, tile_size_1 = 1024)
   
   # 4. prepare to run search
   #    Before calling prepare(), QUBO and m must be given.
   searcher.prepare()

   # 5. run search
   # If a problem is big, search needs long time, proportional to 2^N.
   # search_range() method will make stepwise search,
   # which limits execution time of one call.
   while searcher.search_range() :
      # do your job, such as breaking this loop.
      pass

   # Alternatively, you can use one line for search, searcher.search().
   #
   # searcher.search()   # run whole search.
     
   # 6. make solution
   # calculate QUBO energy and prepare solution as a bit array, and caches them.
   # After calling make_solution(), get_X() and get_E() returns
   # bits and QUBO energy.  
   searcher.make_solution()

   # 7. get solution
   # x is a list of tuples.  Each tuple contains pair of  bit arrays (x0, x1).
   # E is a vector of QUBO energy.
   x = searcher.get_x()
   E = searcher.get_E()



classes and functions
---------------------

sqaod.py.bipartite_graph_bf_searcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.py.bipartite_graph_bf_searcher
		  
sqaod.py.BipartiteGraphBFSearcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.py.BipartiteGraphBFSearcher
   :members:

sqaod.cpu.bipartite_graph_bf_searcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cpu.bipartite_graph_bf_searcher
		  
sqaod.cpu.BipartiteGraphBFSearcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.cpu.BipartiteGraphBFSearcher
   :members:

sqaod.cuda.bipartite_graph_bf_searcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cuda.bipartite_graph_bf_searcher

sqaod.cuda.BipartiteGraphBFSearcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: sqaod.cuda.BipartiteGraphBFSearcher
   :members:
