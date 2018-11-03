Welcome to sqaod documentation!
=================================

Sqaod is a collection of sovlers for simulated quantum annealing, providing a high-performance and stable implementation.
This package is intended for researchers and engineers to explore various problems on quantum computing with conventional workstations and servers. Sqaod is also available for deployment in commercial use-cases.

Features
--------

#. Callable from python
   
   Sqaod supports python 2.7 and python 3.3 and above.

#. QUBO Graphs
    
   Sqaod is capable to deal with two types of graphs of **dense graph** and **bipartite graph**. These graphs have simple mathematical representations, and solved without any modifications.
    
   * Dense graph is the most generic form of QUBO, and utilized for problems such as TSP.
      
   * Bipartite graph is for problems that have input and output nodes in graph. An example is RBM.



#. Algorithm

   Two solver algorithms, **simulated quantum annealing with path-integral monte-carlo** and **brute-force search** are implemented.
    
   * simulated quantum annealing with path-integral monte-carlo is to get approximated solutions for problems with larger number of bits.  One can solve problems with thousands of bits for dense graph and bipartite graph with simulated quantum annealers.
      
   * Brute-force search is for getting strict solutions for problems with smaller number of bits. With brute-force solvers, strict solutions for problems with 30 bits or larger can be obtained within tens of seconds when high-end GPUs are utilized.

#. Reproducibility
    
   One can reproduce results if the same parameters are used.  This is the main difference between numerical simulatio and real quantum hardware.

#. Parallelized and accelerated
    
   Sqaod solvers have **CUDA(NVIDIA-GPU)-based** and **CPU(multicore)-based** backends for acceleration.
    
   * Multi-core CPUs with OpenMP are utilized for CPU-based solvers.
      
   * NVIDIA GPUs by using CUDA are utilized for GPU-based solvers.

#. Problem size and graph complexity
    
   * Since sqaod is a pure software implementation, solvers are able to deal with problems with a large number of bits as long as memory capacity allows.  It also allows flexible graph while real quantum hardware has restrictions on qubit connections.
     Sqaod is also accelerated by modern high-performance devices such as CPUs and GPUs, thus, able to solve problems with large-sized complex graphs.

Examples
--------
Documents for solvers includes simplified psudo code to run solvers.
If you want to run examples, please visit `example at github <https://github.com/shinmorino/sqaod/tree/master/sqaodpy/example>`_.

===============================  ===============  ============================
File name                        graph             algorithm                   
===============================  ===============  ============================
dense_graph_annealer.py_         dense graph      simulated quantum annealing
bipartite_graph_annealer.py_     bipartite graph  simulated quantum annealing
dense_graph_bf_searcher.py_      dense graph      brute-force searcher
bipartite_graph_bf_searcher.py_  bipartite graph  brute-force searcher
===============================  ===============  ============================

.. _dense_graph_annealer.py : https://github.com/shinmorino/sqaod/tree/1.0/sqaodpy/example/dense_graph_annealer.py
.. _dense_graph_bf_searcher.py : https://github.com/shinmorino/sqaod/tree/1.0/sqaodpy/example/dense_graph_bf_searcher.py
.. _bipartite_graph_annealer.py : https://github.com/shinmorino/sqaod/tree/1.0/sqaodpy/example/bipartite_graph_annealer.py
.. _bipartite_graph_bf_searcher.py : https://github.com/shinmorino/sqaod/tree/1.0/sqaodpy/example/bipartite_graph_bf_searcher.py


Table of contents
-----------------
.. toctree::
   :maxdepth: 2

   solvers
	     
   dense_graph_annealer
   bipartite_graph_annealer
   dense_graph_bf_searcher
   bipartite_graph_bf_searcher
   formulas
   
   preference


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
