Dense Graph Brute-force Searcher
================================

Sqaod implements 3 dense graph brute-force searcher classes.
 * `sqaod.py.DenseGraphBFSearcher`_
   
   Python-based reference implementation of dense graph brute-force search.
   
 * `sqaod.cpu.DenseGraphBFSearcher`_
   
   Parallelized CPU implementation by using OpenMP.
   
 * `sqaod.cuda.DenseGraphBFSearcher`_
   
   Parallelized and accelearted by using NVIDIA GPU, CUDA.

Factory function
----------------

Every solver class has its own factory function in the same package.
For example, sqaod.py.DenseGraphBFSearcher is instantiated by `sqaod.py.dense_graph_bf_searcher`_.

Method calling sequence
-----------------------

Below is a psudo-code to run dense graph brute-force searcher.

.. code-block:: python

   import sqaod as sq

   # Assumption
   # N     : problem size.
   # m     : number of trotters.
   # dtype : np.float32. Numerical precision is assumed to be np.float32.
   
   # 1. Preparing QUBO matrix, W.
   #    W is a matrix whose shape is (N, N).
   #    W should be a upper/lower triangular or symmetrix matrix.
   W = create QUBO ...

   # 2. instantiate dense graph brute-force searcher
   searcher = sqaod.py.dense_graph_bf_searcher(W, sq.minimize, dtype)

   # Alternatively, you're able to set QUBO as shown below.
   #
   # searcher = sqaod.py.dense_graph_bf_searcher(dtype)
   # searcher.set_qubo(W, sq.minimize)

   # 3. (optional)
   #    Set preferences. You're able to select tile size.
   #    Preferences are given as **kwargs.
   #
   # searcher.set_preference(tile_size = 1024)
   
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
   x = searcher.get_x()
   # x is a list of tuples.  Each tuple contains pair of  bit arrays (x0, x1).
   # E is a vector of QUBO energy.
   E = searcher.get_E()



classes and functions
---------------------

sqaod.py.dense_graph_bf_searcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.py.dense_graph_bf_searcher
		  
sqaod.py.DenseGraphBFSearcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.py.DenseGraphBFSearcher
   :members:

sqaod.cpu.dense_graph_bf_searcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cpu.dense_graph_bf_searcher
		  
sqaod.cpu.DenseGraphBFSearcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.cpu.DenseGraphBFSearcher
   :members:

sqaod.cuda.dense_graph_bf_searcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cuda.dense_graph_bf_searcher

sqaod.cuda.DenseGraphBFSearcher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: sqaod.cuda.DenseGraphBFSearcher
   :members:
