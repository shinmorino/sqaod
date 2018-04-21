from __future__ import print_function
import numpy as np
from . import checkers
from . import preference as pref
from . import common

class BipartiteGraphBFSearcherBase :
    
    def __init__(self, cext, dtype, b0, b1, W, optimize, prefdict) :
        self._cext = cext
        self.dtype = dtype
        self._optimize = optimize
        if not W is None :
            self.set_qubo(b0, b1, W, optimize)
        self.set_preferences(prefdict)
            
    def __del__(self) :
        self._cext.delete(self._cobj, self.dtype)

    def set_qubo(self, b0, b1, W, optimize = pref.minimize) :
        checkers.bipartite_graph.qubo(b0, b1, W)
        b0, b1, W = common.fix_type([b0, b1, W], self.dtype)
        self._dim = (b0.shape[0], b1.shape[0])
        self._cext.set_qubo(self._cobj, b0, b1, W, optimize, self.dtype)

    def get_problem_size(self) :
        return self._cext.get_problem_size(self._cobj, self.dtype)

    def set_preferences(self, prefdict=None, **prefs) :
        if not prefdict is None :
            self._cext.set_preferences(self._cobj, prefdict, self.dtype)
        self._cext.set_preferences(self._cobj, prefs, self.dtype)

    def get_preferences(self) :
        return self._cext.get_preferences(self._cobj, self.dtype);
        
    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return self._cext.get_E(self._cobj, self.dtype);

    def get_x(self) :
        return self._cext.get_x(self._cobj, self.dtype);
    
    def prepare(self) :
        self._cext.prepare(self._cobj, self.dtype);
        
    def make_solution(self) :
        self._cext.make_solution(self._cobj, self.dtype);
        
    def calculate_E(self) :
        self._cext.calculate_E(self._cobj, self.dtype);
        
    def search_range(self) :
        return self._cext.search_range(self._cobj, self.dtype)
        
    def _search(self) :
        # one liner.  not interruptive.
        self._cext.search(self._cobj, self.dtype);

    def search(self) :
        self.prepare()
        # users is able to use this search loop on his code to get intermediate states
        # such as progress and solutions.
        while True :
            comp, curx0, curx1 = self._cext.search_range(self._cobj, self.dtype)
            if comp :
                break;
        self.make_solution()
