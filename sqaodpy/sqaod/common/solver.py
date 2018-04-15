
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
    def prepare(self) :
        raise NotImplementedError()

    @abstractmethod
    def make_solution(self) :
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
    def prepare(self) :
        raise NotImplementedError()

    @abstractmethod
    def make_solution(self) :
        raise NotImplementedError()

    @abstractmethod
    def calculate_E(self) :
        raise NotImplementedError()

    @abstractmethod
    def anneal_one_step(self, G, beta) :
        raise NotImplementedError()

    
class DenseGraphSolver :
    @abstractmethod
    def get_problem_size(self) : # returns N.
        raise NotImplementedError()

    @abstractmethod
    def set_qubo(self, W, optimize = sqaod.minimize) :
        raise NotImplementedError()

    @abstractmethod
    def get_x(self) : # returns x.
        raise NotImplementedError()

    
class BipartiteGraphSolver :
    @abstractmethod
    def get_problem_size(self) : # returns N0, N1.
        raise NotImplementedError()

    @abstractmethod
    def set_qubo(self, b0, b1, W, optimize = sqaod.minimize) :
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
    def set_hamiltonian(self, h, J, c = 0.) :
        raise NotImplementedError()

    @abstractmethod
    def get_hamiltonian(self, x) :
        raise NotImplementedError()

    @abstractmethod
    def set_q(self, x) :
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
    def set_hamiltonian(self, h0, h1, J, c = 0.) :
        raise NotImplementedError()

    @abstractmethod
    def get_hamiltonian(self, x0, x1) :
        raise NotImplementedError()

    @abstractmethod
    def set_q(self, x0, x1) :
        raise NotImplementedError()

    @abstractmethod
    def get_q(self, x) :
        raise NotImplementedError()

