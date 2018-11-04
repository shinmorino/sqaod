Sqaod solvers
=============

Sqaod is a colletion of solvers for simulated quantum annealing.


QUBO Graphs
-----------
Every solver accepts QUBO as its input, and maximize or minimize energy.  Solvers are able to deal with 2 types of graphs, dense graph and bipartite graph.

Dense graph
^^^^^^^^^^^
Dense grapn is the most general case, and QUBO is defined as show below:

.. math::
   E = \mathbf{x} \mathbf{W} \mathbf{x}^T,

where x is an array of values, { 0, 1 }, and its length is N.  W is a symmetric matrix whose size is (N, N).
Typically QUBOs for TRA are represented as dense graph.

Bipartite graph
^^^^^^^^^^^^^^^
Bipartite graph is a case that has limitted connections between nodes as shown below:

.. math::
   E = \mathbf{b_0} \mathbf{x_0}^T + \mathbf{b_1} \mathbf{x_1}^T +  \mathbf{b_1} \mathbf{W} \mathbf{b_0}^T,

where x0 and x1 are arrays of values, { 0, 1 }, and its length is N0 and N1 respectively.  W is a matrix whose size is (N1, N0).
One notable example is RBM.


Solver algorithm
----------------

Sqaod solvers employ 2 algorithms, **simulated quantum annealing** and **brute-force search**.

Simulated quantum annealing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Solvers are implemented by using path-integral monte-carlo.

(To be filled.)

Brute-force search
^^^^^^^^^^^^^^^^^^

Searching all bit combinations to get maxmum/minimum QUBO energy

(To be filled.)


Devices
-------

Sqaod utilizes CPU and CUDA (with NVIDIA GPU) to accelerate solvers.  Sqaod also has the simple reference implementation for annealing algorithms written in python.

Solvers are implemented in packages of sqaod.py, sqaod.cpu and sqaod.cuda.  Solvers for an algorithm with different devices have the same class interface, so they are exchangeable.


package sqaod.py
^^^^^^^^^^^^^^^^
Python-based solvers for reference implmentation.

This package is aiming at showing solver algorithms.  Algorithms of solvers in other packages are based on those implemented in sqaod.py solvers.  


package sqaod.cpu
^^^^^^^^^^^^^^^^^^^^^^

Solvers are Parallelized and accelerated by using OpenMP.

Please use CPU-accelerated solvers with 1 socket of CPU especially for dense graph annealers.  NUMA is not considered, and execution with multi CPU sockets may degrade performance.  If you're using a multi-socket system, please use taskset or numacto to limit muluti-CPU-socket usage.


package sqaod.cuda
^^^^^^^^^^^^^^^^^^^^^^^

Solvers are parallelized and acclerated by using CUDA

NVIDIA CUDA GPUs are required.  GPUs of compute capability 3.5 (2nd-gen Kepler) or above are supported.  Recommendation is compute capability 5.0(Maxwell) or above.  Please visit `CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`_ at developer.nvidia.com to check compute capabiity of your GPU.

in version 1.0, sqaod uses 1 GPU whose device no is 0, and multi-GPU is not supported.
To choose a GPU, please use CUDA_VISIBLE_DEVICES environement variable.

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=1
   python your-script.py
