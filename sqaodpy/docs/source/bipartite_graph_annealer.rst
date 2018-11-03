Bipartite Graph Annealers
===========================

Sqaod implements 3 bipartite graph annealer classes.
 * `sqaod.py.BipartiteGraphAnnealer`_
   
   Python-based reference implementation of annealing algorithms.
   
 * `sqaod.cpu.BipartiteGraphAnnealer`_
   
   Parallelized CPU implementation by using OpenMP.
   
 * `sqaod.cuda.BipartiteGraphAnnealer`_
   
   Parallelized and accelearted by using NVIDIA GPU, CUDA.

Factory function
----------------

Every solver class has its own factory function in the same package.
For example, sqaod.py.BipartiteGraphAnnealer is instantiated by `sqaod.py.bipartite_graph_annealer`_.

Method calling sequence
-----------------------

Below is a psudo-code to run annealer.

.. code-block:: python

   import sqaod as sq

   # Assumption
   # N0, N1 : problem size.
   # m      : number of trotters.
   # dtype : np.float32. Numerical precision is assumed to be np.float32.
   
   # 1. Preparing QUBO matrix, W.
   #  QUBO energy is defined as E = b0 * x0.T + b1 * x1.T + x1 * W * x0.T
   b0 = ... # vector(length = N0)
   b1 = ... # vector(length = N1)
   W = ...  # matrix, shape == (N1, N0)
   
   # 2. define number of trotters.
   m = number of trotters ...

   # 3. instantiate annaler
   #    The initial value of m is given as m = N / 2.
   ann = sqaod.py.bipartite_graph_annealer(b0, b1, W, m, sq.minimize, dtype)

   # Alternatively, you're able to set W and m as shown below.
   #
   # ann = sqaod.py.bipartite_graph_annealer(dtype)
   # ann.set_qubo(b0, b1, W, sq.minimize)
   # ann.set_preference(n_trotters = m)

   # 4. (optional)
   #    Set preferences. You're able to select annealing algorithm,
   #    number of trotters and so on.
   #    Preferences are given as **kwargs.
   #
   # ann.set_preference(n_trotters = m, algorithm = sqaod.coloring)
   #
   # You're also able to set a seed value for psudo random number generator.
   # ann.set_seed(2391102)
   
   # 5. prepare to run annealing
   #    Before calling prepare(), QUBO and m must be given.
   ann.prepare()

   # 6. randomize spins
   #    Randomize and initialize spins as initial values for annealing.
   ann.randomize_spins()

   # Alternatively, you are able to set spins you generated.
   # Value type of spin should be np.int8.
   #
   # Setting initial spins.  q_init is a vector.
   # q_init = (q0, q1) # array of numpy.int8, len(q0) == N0, len(q1) == N1.
   # ann.set_q(q_init)
   #
   # or
   #
   # Setting a set of initial spins.  The qset_init is a list of tuple
   # storing np.int8 arrays of q0 and q1.
   # qset_init = (q0, q1) # array of numpy.int8, len(q0) == N0, len(q1) == N1.
   # qset_init = [(q00, q10), (q01, q11), ... ]   # len(qset_init) == m
   # ann.set_qset(qset_init)
   
   beta = ...  # set inverse temperature.
   
   # 7. run annealing loop
   while run :  # loop while run == True
     G = ... # updating G for this iteration.

     # 8. anneal.
     #    One call of anneal_one_step() makes tries to flip N x m spins
     ann.anneal_one_step(G, beta)
     
     # 9. (optional) calculate and get QUBO energy if needed.
     #    By calling calculate_E() QUBO energy is calculated and cached.
     #    Execution of calculate_E() is asynchronous for CUDA-based solvers.
     ann.caluculate_E()
     #    E is a vector of QUBO energy for all trotters.
     E = ann.get_E();
     
     # doing something with E.
     
   # 10. make solution
   # calculate QUBO energy and prepare solution as a bit array, and caches them.
   # After calling make_solution(), get_X() and get_E() returns
   # bits and QUBO energy.  
   ann.make_solution()

   # 11. get solution
   x = ann.get_x()  # list of tuples, x0 and x1.
                    #  x0 and x1 are bit arrays storing solution
   E = ann.get_E()
   

classes and functions
---------------------

sqaod.py.bipartite_graph_annealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.py.bipartite_graph_annealer
		  
sqaod.py.BipartiteGraphAnnealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.py.BipartiteGraphAnnealer
   :members:

sqaod.cpu.bipartite_graph_annealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cpu.bipartite_graph_annealer
		  
sqaod.cpu.BipartiteGraphAnnealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.cpu.BipartiteGraphAnnealer
   :members:

sqaod.cuda.bipartite_graph_annealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cuda.bipartite_graph_annealer

sqaod.cuda.BipartiteGraphAnnealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: sqaod.cuda.BipartiteGraphAnnealer
   :members:
