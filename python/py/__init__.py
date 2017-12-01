import solver_traits

from dense_graph_annealer import dense_graph_annealer
from rbm_annealer import rbm_annealer
from rbm_bf_solver import rbm_bf_solver

def create(selector, dict = None) :
    return dense_graph_annealer()
