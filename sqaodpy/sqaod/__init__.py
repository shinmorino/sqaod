# consts for factory methods

# graph type   FIXME: give correct definitions

dense = 0
bipartite = 1

# solver type

bruteforce = 0
anneal = 1

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
