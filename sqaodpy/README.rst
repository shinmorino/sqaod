=====
Sqaod
=====

Sqaod is a collection of sovlers for simulated quantum annealing, providing a high-performant and stable implementation to simulate quantum annealing.

This package is intended for researchers and engineers  to explore various problems on qunatum computing with conventional workstations and servers.  Sqaod is also available for deployment in commercial usecases.
Please visit `sqaod website <https://github.com/shinmorino/sqaod>`_ and `sqaod wiki <https://github.com/shinmorino/sqaod/wiki>`_ at github for details.

Installation
------------
In order to use sqaod, you also need to install native libraries.  Please visit `Installation <https://github.com/shinmorino/sqaod/wiki/Installation>`_ for details.


Features
--------

* Solving annealing problems with simple mathmatical definitions. 
  
  Sqaod is capable to deal with two graphs of **dense graph** and **bipartite graph**.  These graphs have simple mathmatical representations, and directly solved without any modifications.
  
  * Dense graph is the most generic form of QUBO, and utilized for problems such as TSP.
  
  * Bipartite graph is for problems that have input and output nodes in graph.  An example is RBM.  

* Two solver algorithm, **brute-force search** and **monte-carlo-based simulated quantum annealer** are implemented.
  
  * Monte-carlo based simulated quantum annealer is to get approximated solutions for problems with larger number of bits.|br| 
    One can solve problems with thousands of bits for dense graph and bipartite graph with simulated quantum anneaers.

  * Brute-force search is for getting strict solutions for problems with smaller number of bits.
    With brute-force solvers, strict solutions for 30-bit Problem are able to be obtained within tens of seconds when high-end GPUs are utilized.
    
  
* Acceerated on CPU and GPU.
  
  Sqaod solvers have C++- and CUDA-based backends for acceleration.
  
  * Multi-core CPUs with OpenMP are utilized for CPU-based solvers.
  * NVIDIA GPUs by using CUDA are utilized for GPU-based solvers.
  
* Able to solve problems with large number of bits.

  Sqaod is a software implementation for simulated quantum annealing.  Solvers are able to deal with problems with a large number of bits, while other hardware devices have limitation on solving large problems.

  Problem sizes are limited by memory amount and/or calculation time.  On recent workstations and servers large amount of DRAM are available, and performance of Sqaod is excellent since it's optimized on modern computing devices.
  
Release history
---------------

Current version is 1.0.2.

* Ver 1.0.2
  
  * Version 1.0.2 includes miscellaneous bug fixes, that affect annealing behavior.   Please update to 1.0.2 if you're using older versions.

  * getSystemE() is added to solvers to calculate system energy during annealing. [#60]

  * sqaod.algorithm.sa_default is added to select default SA algorithms in annealers. [#61]

  * calculate_E() and make_solutions() are not required to get QUBO energy and solutions.  These functions are for caching energies and solutions. [#63]

  * Python solvers return copies of objects.[#62]

  * Fix: anneal_one_step() for SA algorithm did not work, since parameters are not correctly passed. [#65]

  * Fix: QUBO energy was not correctly calculated and beta was not correctly applied in SQA algorithms. [#64]

  * Fix: symmetrize() was not correctly handled. [#66]


* Ver 1.0.1

  * `Documentation <https://shinmorino.github.io/sqaod/docs/1.0>`_ prepared.

  * Updated some version sinagures that were not updated.

* Ver 1.0.0

  * All solvers and functions are able to accept upper/lower triangular matrices. `[#57] <https://github.com/shinmorino/sqaod/issues/57>`_.

  * Simulated annealing algorithms (not simulated *quantum* annealing) have been implemented.  It's automatically selected when n_trotters == 1.  `[#59] <https://github.com/shinmorino/sqaod/issues/59>`_.

  * Misc bug fixes.

* Ver 0.3.1 (Beta2 Update1)

  * No changes in solvers and programming interface
  * Adding environmental checks(library installation, versions).

* Ver 0.3.0 (Beta2)

  * Python interfaces are fixed, and most functionalities are tested.
  * Remaining works are optimizations and documentation, which are going to be made by Beta2 planned in the end of June.
