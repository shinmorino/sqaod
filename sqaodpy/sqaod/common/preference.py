# operations to switch minimize / maximize.

class Algorithm :
    pass

algorithm = Algorithm()
algorithm.default = 'default'
algorithm.naive = 'naive'
algorithm.coloring = 'coloring'
algorithm.brute_force_search = 'brute_force_search'
algorithm.sa_naive = 'sa_naive'
algorithm.sa_coloring = 'sa_coloring'


class Minimize :
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
    def __int__(self) :
        return 1

minimize = Minimize()
maximize = Maximize()
