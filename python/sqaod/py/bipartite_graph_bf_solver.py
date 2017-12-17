import numpy as np
import random
import sys
import utils
import tags

class BipartiteGraphBFSolver :
    
    def __init__(self, W, b0, b1, optimize) :
        self._verbose = False
        if not W is None :
            self.set_problem(W, b0, b1, optimize)

    def _vars(self) :
        return self._W, self._blist[0], self._blist[1], self._xlist[0], self._xlist[1]

    def _Esign(self) :
        return self._optimize.Esign

    def _get_dim(self) :
        return self._dim[0], self._dim[1]

    def _check_dim(self, W, b0, b1) :
        N0 = W.shape[1]
        N1 = W.shape[0]
        if len(b0) != N0 :
            return False;
        if len(b1) != N1 :
            return False;
        return True
    
    def set_problem(self, W, b0, b1, optimize = tags.minimize) :
        if not self._check_dim(W, b0, b1) :
            raise Exception('dimension does not match, W: {0}, b0: {1}, b1: {2}.'\
                            .format(W.shape, b0.size, b1.size))
        self._dim = (b0.shape[0], b1.shape[0])
        N0, N1 = self._get_dim()
        self._xlist = ( np.zeros((N0), dtype=np.int8), np.zeros((N1), dtype=np.int8) )
        self._optimize = optimize
        self._W = self._Esign() * W
        self._blist = (self._Esign() * b0, self._Esign() * b1)
        

    def get_E(self) :
        return self._Esign() * self._Emin

    def get_solutions(self) :
        return self._solutions

    def _reset_solutions(self, Etmp, x0, x1) :
        self._solutions = [(self._Esign() * Etmp, x0, x1)]
        if self._verbose :
            print("Enew < E : {0}".format(Etmp))

    def _append_to_solutions(self, Etmp, x0, x1) :
        self._solutions.append((self._Esign() * Etmp, x0, x1))
        if self._verbose :
            print("Etmp < Emin : {0}".format(Etmp))
    
    def _search(self) :
        N0, N1 = self._get_dim()
        iMax = 1 << N0
        jMax = 1 << N1
        self._Emin = sys.float_info.max
        W, b0, b1, x0, x1 = self._vars()

        for i in range(iMax) :
            x0 = utils.create_bits_sequence((i), N0)
            for j in range(jMax) :
                x1 = utils.create_bits_sequence((j), N1)
                Etmp = np.dot(b0, x0.transpose()) + np.dot(b1, x1.transpose()) \
                       + np.dot(x1, np.matmul(W, x0.transpose()))
                if self._Emin < Etmp :
                    continue
                elif Etmp < self._Emin :
                    self._Emin = Etmp
                    self._reset_solutions(Etmp, x0, x1)
                else :
                    self._append_to_solutions(Etmp, x0, x1)
        
    def _search_batched(self) :
        N0, N1 = self._get_dim()
        iMax = 1 << N0
        jMax = 1 << N1
        self._Emin = sys.float_info.max
        W, b0, b1, _, _ = self._vars()

        iStep = min(256, iMax)
        jStep = min(256, jMax)
        for iTile in range(0, iMax, iStep) :
            x0 = utils.create_bits_sequence(range(iTile, iTile + iStep), N0)
            for jTile in range(0, jMax, jStep) :
                x1 = utils.create_bits_sequence(range(jTile, jTile + jStep), N1)
                Etmp = np.matmul(b0, x0.T).reshape(1, iStep) \
                       + np.matmul(b1, x1.T).reshape(jStep, 1) + np.matmul(x1, np.matmul(W, x0.T))
                for j in range(jStep) :
                    for i in range(iStep) :
                        if self._Emin < Etmp[j][i] :
                            continue
                        elif Etmp[j][i] < self._Emin :
                            self._Emin = Etmp[j][i]
                            self._reset_solutions(Etmp[j][i], x0[i], x1[j])
                        else :
                            self._append_to_solutions.append(Etmp[j][i], x0[i], x1[j])
                    

    def search(self) :
        # self._search_optimum()
        self._search_batched()
        

def bipartite_graph_bf_solver(W = None, b0 = None, b1 = None, optimize = tags.minimize) :
    return BipartiteGraphBFSolver(W, b0, b1, optimize)


if __name__ == '__main__' :
    N0 = 14
    N1 = 5
    
    np.random.seed(0)

    W = np.random.random((N1, N0)) - 0.5
    b0 = np.random.random((N0)) - 0.5
    b1 = np.random.random((N1)) - 0.5
    
    bf = bipartite_graph_bf_solver(W, b0, b1)
    bf._search()
    E = bf.get_E()
    solutions = bf.get_solutions() 
    print(E, solutions)
    
    bf._search_batched()
    E = bf.get_E()
    solutions = bf.get_solutions() 
    print(E, solutions)

