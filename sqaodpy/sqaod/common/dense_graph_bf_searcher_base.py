from __future__ import print_function
import numpy as np
from . import checkers
from . import preference as pref
from . import common

class DenseGraphBFSearcherBase :
    
    def __init__(self, cext, dtype, W, optimize, prefdict) :
        self._cext = cext
        self.dtype = dtype
        if not W is None :
            self.set_qubo(W, optimize)
        self.set_preferences(prefdict)
            
    def __del__(self) :
        if hasattr(self, '_cobj') :
            self._cext.delete(self._cobj, self.dtype)

    def set_qubo(self, W, optimize = pref.minimize) :
        checkers.dense_graph.qubo(W)
        W = common.fix_type(W, self.dtype)
        W = common.symmetrize(W)
        self._N = W.shape[0]
        self._cext.set_qubo(self._cobj, W, optimize, self.dtype)
        self._optimize = optimize

    def get_problem_size(self) :
        return self._cext.get_problem_size(self._cobj, self.dtype)

    def set_preferences(self, prefdict = None, **prefs) :
        if not prefdict is None :
            self._cext.set_preferences(self._cobj, prefdict, self.dtype)
        self._cext.set_preferences(self._cobj, prefs, self.dtype)

    def get_preferences(self) :
        return self._cext.get_preferences(self._cobj, self.dtype);

    def get_optimize_dir(self) :
        return self._optimize

    def get_E(self) :
        return self._cext.get_E(self._cobj, self.dtype)

    def get_x(self) :
        return self._cext.get_x(self._cobj, self.dtype)

    def prepare(self) :
        self._cext.prepare(self._cobj, self.dtype);
        
    def make_solution(self) :
        self._cext.make_solution(self._cobj, self.dtype)

    def calculate_E(self) :
        self._cext.calculate_E(self._cobj, self.dtype)
        
    def search_range(self) :
        return self._cext.search_range(self._cobj, self.dtype)
        
    def search(self) :
        self.prepare()
        while True :
            comp, curx = self._cext.search_range(self._cobj, self.dtype)
            if comp :
                break;
        self.make_solution()
        
    def _search(self) :
        # one liner.  does not accept ctrl+c.
        self._cext.search(self._cobj, self.dtype)


