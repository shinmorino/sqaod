Preferences
===========

Solvers in sqaod has preferences, which are defined as values with keywords.  In v1.0.x, following preferences are defined.

+---------------+--------+-------------------+---------------------------------------------------------+
| keyword       | access | type              | description                                             |
+===============+========+===================+=========================================================+
| **common preferences**                                                                               |
+---------------+--------+-------------------+---------------------------------------------------------+
| algorithm_    |  R/W   | str               | select algorithm.                                       |
+---------------+--------+-------------------+---------------------------------------------------------+
| precision_    |  RO    | str               | numerical precision (double/float)                      |
+---------------+--------+-------------------+---------------------------------------------------------+
| device_       |  RO    | str               | computing device an annlaer is using                    |
+---------------+--------+-------------------+---------------------------------------------------------+
| **preference available on annealers**                                                                |
+---------------+--------+-------------------+---------------------------------------------------------+
| n_trotters_   | R/W    |  positive integer | number of trotters for annealers.                       |
+---------------+--------+-------------------+---------------------------------------------------------+
| **preference available on brute-force searcher**                                                     |
+---------------+--------+-------------------+---------------------------------------------------------+
| tile_size_    | R/W    |  positive integer | tile size for dense graph brute-force searchers         |
+---------------+--------+-------------------+---------------------------------------------------------+
| tile_size_0_  | R/W    |  positive integer | tile size 0 for bipartite graph brute-force searchers   |
+---------------+--------+-------------------+---------------------------------------------------------+
| tile_size_1_  | R/W    |  positive integer | tile size for bipartite graph brute-force searchers     |
+---------------+--------+-------------------+---------------------------------------------------------+


algorithm
--------------------------
**algorithm** is a R/W preference to select sovler algorithm.
You can select algorithm by passing sqaod.algorithm_ to annealers' set_preference() method.

ex.

.. code-block:: python

   import sqaod as sq
   import sqoad.algorithm as algo
   
   W = create QUBO ...

   # create a dense graph annealer to minimize QUBO(W) energy.
   ann = sq.cuda.dense_graph_annealer(W, sqaod.minimize)
   
   # select algorithm.
   ann.set_preferences(algorithm = algo.coloring)

.. note::
   * Each solver has its default SQA algorithm and SA algorithm.  If you set sqaod.default as algorith, an annealer selects its default one.
   * For cases of n_trotters == 1, an annealer will select its SA default algorithms because SQA algorithms do not work in these cases.
   * If an user selects an unsupported algorithm, solver will select its default algorithm, and log it.
  		

Dense Graph Annealer Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annealres in sqaod implements algorithms shown below.

+-----------------------------------+---------+----------+----------+-------------+
| Annealer class                    |  naive  | coloring | sa_naive | sa_coloring |
+===================================+=========+==========+==========+=============+
| sqaod.py.DenseGraphAnnealer       |    D    |    x     |    x     |             |
+-----------------------------------+---------+----------+----------+-------------+
| sqaod.cpu.DenseGraphAnnealer      |    x    |    D     |    d     |             |
+-----------------------------------+---------+----------+----------+-------------+
| sqaod.cuda.DenseGraphAnnealer     |         |    D     |    d     |             |
+-----------------------------------+---------+----------+----------+-------------+
| sqaod.py.BipartiteGraphAnnealer   |    D    |    x     |    D     |      x      |
+-----------------------------------+---------+----------+----------+-------------+
| sqaod.cpu.BipartiteGraphAnnealer  |    x    |    D     |    x     |      d      |
+-----------------------------------+---------+----------+----------+-------------+
| sqaod.cuda.BipartiteGraphAnnealer |         |    D     |          |      d      |
+-----------------------------------+---------+----------+----------+-------------+

D = default SQA algorithm, d = default SA algorithm, x = selectable.


Bipartite Graph Annealer Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The only algorithm that brute-force searcher has **sqaod.algorithm.brute_force_search**.

Users are not able to select any other algorithms.  If you try to select other algorithms, brute-force searchers simplly ignore it.


sqaod.algorithm
^^^^^^^^^^^^^^^
Module level attributes, **sqaod.algorithm**, is used to select solver algorithm.  **sqaod.algorithm** is defined in sqaod.common, and its atribute is accessible as **sqaod.algorithm.default** as an example.

.. automodule:: sqaod.common.preference
   :members: algorithm


precision
^^^^^^^^^

**precision** is a read-only attribute to show solver numerical precision.

Value
  "float" or "double".

.. note::
   Solvers in sqaod.py package does not return precision.


device
^^^^^^

**device** is a read-only preference, representing device type used by solvers.

Value
  'CPU' for solvers in sqaod.cpu package
  
  'CUDA' for solvers in sqaod.cuda package.

.. note::
   Solvers in sqaod.py package does not return device.

n_trotters
^^^^^^^^^^

**n_trotters** is a read-write preference available in annealers.

Value
  positive integer.

For SQA algorithms, n_trotters works to specify number of trotters as the name tells.
For SA algorithms, n_trotters works as number of qubit arrays parallelly annealed.

tile_size
^^^^^^^^^

**tile_size** is a read-write preference available in dense graph brute-force searchers.

Value
  positive integer.

Brute-force searchers do batched calculation of QUBO energy, and the tile_size preference specifies batch size.

Typically larger tile_size will give better performance.

Some brute-force searchers may correct given values to more optimal ones.

.. _tile_size_0: #tile-size-0-tile-size-1

.. _tile_size_1: #tile-size-0-tile-size-1

tile_size_0, tile_size_1
^^^^^^^^^^^^^^^^^^^^^^^^

The **tile_size_0** and **tile_size_1** are read-write preferences available in bipartite graph brute-force searchers.

Value
  positive integer.

Brute-force searchers do batched calculation of QUBO energy, and the tile_size_0 and tile_size_1 preferences specifie batch size.

Typically larger tile_size will give better performance.

Some brute-force searchers may correct given values to more optimal ones.


sqaod.maximize, sqaod.minimize
------------------------------
Module level attributes, **sqaod.maximize** and **sqaod.minimize** are used to specify optimize direction to maximize or minimize QUBO energy. Though these attributes are originally defined in sqaod.common.preference module, they're imported directly under sqaod package for ease of use.
   
.. autoattribute:: sqaod.common.preference.maximize

.. autoattribute:: sqaod.common.preference.minimize
