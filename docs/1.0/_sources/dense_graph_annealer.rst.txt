Dense Graph Annealers
===========================

Sqaod implements 3 dense graph annealer classes.
 * `sqaod.py.DenseGraphAnnealer`_
   
   Python-based reference implementation of annealing algorithms.
   
 * `sqaod.cpu.DenseGraphAnnealer`_
   
   Parallelized CPU implementation by using OpenMP.
   
 * `sqaod.cuda.DenseGraphAnnealer`_
   
   Parallelized and accelearted by using NVIDIA GPU, CUDA.

Factory function
----------------

Every solver class has its own factory function in the same package.
For example, sqaod.py.DenseGraphAnnealer is instantiated by `sqaod.py.dense_graph_annealer`_.

Method calling sequence
-----------------------

Below is a psudo-code to run annealer.

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
   
   # 2. define number of trotters.
   m = number of trotters ...

   # 3. instantiate annaler
   #    The initial value of m is given as m = N / 2.
   ann = sqaod.py.dense_graph_annealer(W, m, sq.minimize, dtype)

   # Alternatively, you're able to set W and m as shown below.
   #
   # ann = sqaod.py.dense_graph_annealer(dtype)
   # ann.set_qubo(W, sq.minimize)
   # ann.set_preference(n_trotters = m)

   # 4. (optional)
   #    Set preferences. You're able to select annealing algorithm, number of trotters and so on.
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
   # q_init = <array of numpy.int8, len(q_init) == N>
   # ann.set_q(q_init)
   #
   # or
   #
   # Setting a set of initial spins.  qset_init is a list of numpy.int8 vector.
   # qset_init = [q0, q1, ... ]   # len(qset_init) == m
   # ann.set_qset(qset_init)
   
   beta = ...  # set inverse temperature.
   
   # 7. run annealing loop
   while run :  # loop while run == True
     G = ... # updating G for this iteration.

     # 8. anneal.
     #    One call of anneal_one_step() makes tries to flip N x m spins
     ann.anneal_one_step(G, beta)
     
     # 9. (optional) calculate and get QUBO energy if needed.
     #    E is a vector of QUBO energy for all trotters.
     E = ann.get_E();
     
     # doing something with E.
     
   # 10. get solution
   x = ann.get_x()
   E = ann.get_E()


classes and functions
---------------------

sqaod.py.dense_graph_annealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.py.dense_graph_annealer
		  
sqaod.py.DenseGraphAnnealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.py.DenseGraphAnnealer
   :members:

sqaod.cpu.dense_graph_annealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cpu.dense_graph_annealer
		  
sqaod.cpu.DenseGraphAnnealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sqaod.cpu.DenseGraphAnnealer
   :members:

sqaod.cuda.dense_graph_annealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sqaod.cuda.dense_graph_annealer

sqaod.cuda.DenseGraphAnnealer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: sqaod.cuda.DenseGraphAnnealer
   :members:
