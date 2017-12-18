# consts for factory methods

# graph type

dense = 0
sparse = 1
rbm = 2

# solver type

bruteforce = 0
anneal = 1

# optimize type

class Minimize :
    Esign = 1.
    def __trunc__(self) :
        return 0

class Maximize :
    Esign = -1.
    def __trunc__(self) :
        return 1

minimize = Minimize()
maximize = Maximize()


# imports
import common
import py
