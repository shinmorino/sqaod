Formulas
========

Formulas modules implement common and shared functions for solvers such as
 * QUBO -> Hamiltonian conversion
 * Energy calculation frmm bits or spins including batched versions.

There are 3 versions of formulas modules in sqaod.py, sqaod.cpu and sqaod.cuda.

The sqoad.py.formulas module has reference implementations.  Other formuas modules in sqaod.cpu and sqaod.cuda are accleretaed by CPU or CUDA respectively.


sqaod.py.formulas
^^^^^^^^^^^^^^^^^

:note: A parameter, dtype, has None as the default value, which is simply ignored in sqaod.py.formulas.
       Dtype paramter here is to keep function sigunatures being the same as those in other modules.

.. automodule:: sqaod.py.formulas
   :members:


sqaod.cpu.formulas, sqaod.cuda.formulas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions in sqaod.cpu.formulas sqaod.cuda.formulas have the same sinatures and functionlities, thus, documentations are ommited here.

One exception is the dtype parameter.   The dtype paramter does not have its default value, and should be explicitly specified.  Allowed values are numpy.float32 and numpy.float64.  Floating-point parameters (such as vector, matrix, scalars) given to these functions will be converted to the type specified by dtype before calculation.
