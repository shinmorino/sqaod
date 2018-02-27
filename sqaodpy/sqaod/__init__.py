
class Algorithm :
    pass

algorithm = Algorithm()
setattr(algorithm, 'default', 'default')
setattr(algorithm, 'naive', 'naive')
setattr(algorithm, 'cloring', 'coloring')

algorithm.default = 'default'
algorithm.naive = 'naive'
algorithm.coloring = 'coloring'
algorithm.brute_force_search = 'brute_force_search'


# operations to switch minimize / maximize.

class Minimize :
    @staticmethod
    def sign(v) :
        return v
    @staticmethod
    def best(list) :
        return min(list)
    @staticmethod
    def sort(list) :
        return sorted(list)
    def __trunc__(self) :
        return 0

    
class Maximize :
    @staticmethod
    def sign(v) :
        return -v
    @staticmethod
    def best(list) :
        return max(list)
    @staticmethod
    def sort(list) :
        return sorted(list, reverse = true)
    def __trunc__(self) :
        return 1

minimize = Minimize()
maximize = Maximize()


# imports
from sqaod.common import *
import sqaod.py
import sqaod.cpu
from sqaod.common.cuda import is_cuda_available
