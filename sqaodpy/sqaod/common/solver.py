
class Solver :
    
    @abstractmethod
    def select_algorithm(self, algo) :
        raise NotImplementedError()

    @abstractmethod
    def get_preferences(self, **kwargs) :
        raise NotImplementedError()

    @abstractmethod
    def set_preferences(self, **kwargs) :
        raise NotImplementedError()

    @abstractmethod
    def get_E(self) :
        raise NotImplementedError()

class BFSearcher(Solver) :

    @abstractmethod
    def init_search(self) :
        raise NotImplementedError()

    @abstractmethod
    def fin_search(self) :
        raise NotImplementedError()

    @abstractmethod
    def search(self) :
        raise NotImplementedError()

    
class Annealer(Sovler) :
    
    @abstractmethod
    def seed(self, seed) :
        raise NotImplementedError()

    @abstractmethod
    def randomize_spin(self) :
        raise NotImplementedError()

    @abstractmethod
    def init_anneal(self) :
        raise NotImplementedError()

    @abstractmethod
    def fin_anneal(self) :
        raise NotImplementedError()

    @abstractmethod
    def calculate_E(self) :
        raise NotImplementedError()

    @abstractmethod
    def anneal_one_step(self, G, kT) :
        raise NotImplementedError()

    
class DenseGraphSolver :
    @abstractmethod
    def get_problem_size(self) : # returns N.
        raise NotImplementedError()

    @abstractmethod
    def set_problem(self, W, optimize = sqaod.minimize) :
        raise NotImplementedError()

    @abstractmethod
    def get_x(self) : # returns x.
        raise NotImplementedError()

    
class BipartiteGraphSolver :
    @abstractmethod
    def get_problem_size(self) : # returns N0, N1.
        raise NotImplementedError()

    @abstractmethod
    def set_problem(self, b0, b1, W, optimize = sqaod.minimize) :
        raise NotImplementedError()

    @abstractmethod
    def get_x(self) : # returns x0 and x1.
        raise NotImplementedError()
                             

class DenseGraphBFSearcher(BFSearcher, DenseGraphSolver) :
    @abstractmethod
    def search_range(self, xBegin, xEnd) :
        raise NotImplementedError()

    def search(self) : # to be implemented.
        raise NotImplementedError()
    
    
class DenseGraphAnnealer(Annealer, DenseGraphSolver) :
    @abstractmethod
    def get_hJc(self, x) :
        raise NotImplementedError()

    @abstractmethod
    def set_x(self, x) :
        raise NotImplementedError()

    @abstractmethod
    def get_q(self, x) :
        raise NotImplementedError()

    
class BipartiteGraphBFSearcher(BFSearcher, BipartiteGraphSolver) :
    @abstractmethod
    def search_range(self, x0Begin, x0End, x1Begin, x1End) :
        raise NotImplementedError()

    def search(self) : # to be implemented.
        raise NotImplementedError()
    
class BipartiteGraphAnnealer(Annealer, BipartiteGraphSolver) :
    @abstractmethod
    def get_hJc(self, x0, x1) :
        raise NotImplementedError()

    @abstractmethod
    def set_x(self, x0, x1) :
        raise NotImplementedError()

    @abstractmethod
    def get_q(self, x) :
        raise NotImplementedError()

