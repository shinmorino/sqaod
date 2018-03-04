import numpy as np
import sqaod as sq
    

N0 = 8
N1 = 8

# 1. set problem.  As an example, a matrix filled with 1. is used.
b0 = np.ones(N0)
b1 = np.ones(N1)
W = np.ones((N1, N0))

# 2. choosing solver .
sol = sq.cpu # use CPU annealer
# If you want to use CUDA, check CUDA availability with sq.is_cuda_available().
if sq.is_cuda_available() :
    import sqaod.cuda
    sol = sqaod.cuda
    
# 3. instanciate solver
ann = sol.bipartite_graph_annealer()

# 4. (optional) set random seed.
ann.seed(13255)

# 5. Setting problem
# Setting W and optimize direction (minimize or maxminize)
# n_trotters is implicitly set to N/4 by dfault.
ann.set_problem(b0, b1, W, sq.maximize)

# 6. set preferences,
# The following line set n_trotters is identical to the dimension of W.
ann.set_preferences(n_trotters = N0 + N1)

# altternative for 4. and 5.
# W, optimize direction and n_trotters are able to be set on instantiation.

# ann = sol.dense_graph_annealer(b0, b1, W, sq.minimize, n_trotters = W.shape[0])

# 7. get ising model paramters. (optional)
# When W and optimize dir are set, ising model parameters of h, J and c are caluclated.
# By using get_hJc() to get these values. 
h0, h1, J, c = ann.get_hJc()
print 'h0=', h0
print 'h1=', h1
print 'J=', J
print 'c=', c

# 8. showing preferences (optional)
# preferences of solvers are obtained by calling get_preference().
# preferences is always repeseted as python dictionay object.
print ann.get_preferences()

# 9. initialize anneal to setup solver. Annealers must be initialized
#  before calling randomize_q() and/or anneal_one_step().
ann.init_anneal()

# 10. randomize or set x(0 or 1) to set the initial state (mandatory)
ann.randomize_q()

# 11. annealing

Ginit = 5.
Gfin = 0.01
kT = 0.02
tau = 0.99

# annealing loop
G = Ginit
while Gfin <= G :
    # 11. call anneal_one_step to try fliping bits for (n_bits x n_trotters) times.
    ann.anneal_one_step(G, kT)
    G *= tau

# 12. finalize anneal to calculate E and get x(0, 1) from q(-1, 1) internally.
ann.fin_anneal()

# 13. some methods to get results
# - Get list of E for every trogger.
E = ann.get_E()
# - Get annealed q. get_q() returns q matrix as (n_trotters, N)
qlist = ann.get_q()
# - Get annealed x. get_x() returns x matrix as (n_trotters, N)
xlist = ann.get_x()


# 14. creating summary object
summary = sq.make_summary(ann)

# 15. get the best engergy(for min E for minimizing problem, and max E for maxmizing problem)
print 'E {}'.format(summary.E)

# 16. show the number of solutions that has the same energy of the best E.
print 'Number of solutions : {}'.format(len(summary.xlist))

# 17. show solutions. Max number of x is limited to 4.
nToShow = min(len(summary.xlist), 4)
for idx in range(nToShow) :
    print summary.xlist[idx]
