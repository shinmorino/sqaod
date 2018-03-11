import numpy as np


def sort_bitset_list(bitsSetList) :
    size = len(bitsSetList)
    nBitsSetList = len(bitsSetList[0])
    for bitsIdx in range(nBitsSetList) :
        N = bitsSetList[0][bitsIdx].shape[0]
        for pos in range(N - 1, -1, -1) :
            sets0, sets1 = [], []
            for idx in range(size) :
                if bitsSetList[idx][bitsIdx][pos] == 0 :
                    sets0.append(bitsSetList[idx])
                else :
                    sets1.append(bitsSetList[idx])
        bitsSetList = sets0 + sets1

    return bitsSetList


def sort_bitset(bitsSetList) :
    if type(bitsSetList[0]) is tuple or type(bitsSetList[0]) is list :
        return sort_bitset_list(bitsSetList)

    size = len(bitsSetList)
    nBitsSetList = len(bitsSetList)
    N = bitsSetList[0].shape[0]
    for pos in range(N - 1, -1, -1) :
        sets0, sets1 = [], []
        for idx in range(size) :
            if bitsSetList[idx][pos] == 0 :
                sets0.append(bitsSetList[idx])
            else :
                sets1.append(bitsSetList[idx])
        bitsSetList = sets0 + sets1
    return bitsSetList

def unique_x(x) :
    x = sort_bitset(x)
    u = []
    while len(x) != 0 :
        x0 = x[0]
        x.pop(0)
        u.append(x0)
        while len(x) != 0 and np.allclose(x0, x[0]) :
            x.pop(0)
            
    return u


class Summary :
    def __init__(self, solver = None) :
        if not solver is None :
            self.summarize(solver)

    def summarize(self, solver) :
        E = solver.get_E()
        xset = solver.get_x()

        optdir = solver.get_optimize_dir();
        self.E = optdir.best(E)
        bestX = []
        for idx in range(E.shape[0]) :
            if E[idx] == self.E :
                bestX.append(xset[idx])
                
        self.xlist = unique_x(bestX)

    
def make_summary(solver = None) :
    return Summary(solver)

