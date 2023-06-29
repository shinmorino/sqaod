# operations to switch minimize / maximize.

class Algorithm :
    default = 'default'
    naive = 'naive'
    coloring = 'coloring'
    brute_force_search = 'brute_force_search'
    sa_default = 'sa_default'
    sa_naive = 'sa_naive'
    sa_coloring = 'sa_coloring'

    @staticmethod
    def is_sqa(algo) :
        if algo == Algorithm.default or \
           algo == Algorithm.naive or \
           algo == Algorithm.coloring :
            return True
       
        return False
   
    
algorithm = Algorithm()
"""
  sqaod.common.Algorithm: signatures to select algorithms

  algorithm signatures are provided as attributes shown below:

  **default:**
    select default algorithm.
    Each solver class has its default algorithm.  By specifying algorithm.default, 
    the default algirhtm will be chosen,
  **naive :**
    Naive version of SQA algorithm,
  **coloring :**
    SQA algorithm is parallelized by using color-coding,
  **brute_force_search :**
    Brute-force search.
    All brute-force searchers have 'brute_force_search' algorithm,
  **sa_default :**
    select default algorithm for SA(simulated annealing, not SQA),
  **sa_naive :**
    naive version of SA,
  **sa_coloring :**
    color-code version of SA.

"""



class Minimize :
    """Tag for optimize direction.  This class also has some methods for sign manipulations."""
    @staticmethod
    def sign(v) :
        return v.copy()
    @staticmethod
    def best(list) :
        return min(list)
    @staticmethod
    def sort(list) :
        return sorted(list)
    def __int__(self) :
        return 0
    __index__ = __int__

    
class Maximize :
    """Tag for optimize direction.  This class also has some methods for sign manipulations."""
    @staticmethod
    def sign(v) :
        return -v
    @staticmethod
    def best(list) :
        return max(list)
    @staticmethod
    def sort(list) :
        return sorted(list, reverse = true)
    def __int__(self) :
        return 1
    __index__ = __int__

minimize = Minimize()  #: telling solvers to minimize QUBO energy.
maximize = Maximize()  #: telling solvers to maximize QUBO energy.
