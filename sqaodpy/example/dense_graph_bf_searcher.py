import numpy as np
import sqaod as sq
    

N = 8

# 1. set problem.  As an example, a matrix filled with 1. is used.
W = np.ones((N, N))

# 2. choosing solver .
sol = sq.cpu # use CPU searchealer
# If you want to use CUDA, check CUDA availability with sq.is_cuda_available().
if sq.is_cuda_available() :
    import sqaod.cuda
    sol = sqaod.cuda
    
# 3. instanciate solver
search = sol.dense_graph_bf_searcher()

# 4. Setting problem
# Setting W and optimize direction (minimize or maxminize)
# n_trotters is implicitly set to N/4 by dfault.
search.set_qubo(W, sq.maximize)

# 5. (optional) set preferences,
# Set tile_size.  Typical not required.
search.set_preferences(tile_size = W.shape[0])

# altternative for 4. and 5.
# W, optimize direction and tile_size are able to be set on instantiation.

# search = sol.dense_graph_searchealer(W, sq.minimize, tile_size = W.shape[0])

# 6. showing preferences (optional)
# preferences of solvers are obtained by calling get_preference().
# preferences is always repeseted as python dictionay object.
print search.get_preferences()

# 7. do brute-force search
# during search, x vectors that has minimum E values at the point of scan are saved.
search.search()

# 8. some methods to get results
# - Get list of E for every solution.
E = search.get_E()
# - Get searchealed x. get_x() returns x matrix as (n_trotters, N)
x = search.get_x()


# 9. creating summary object
summary = sq.make_summary(search)

# 10. get the best engergy(for min E for minimizing problem, and max E for maxmizing problem)
print 'E {}'.format(summary.E)

# 11. show the number of solutions that has the same energy of the best E.
print 'Number of solutions : {}'.format(len(summary.xlist))

# 12. show solutions. Max number of x is limited to 4.
nToShow = min(len(summary.xlist), 4)
for idx in range(nToShow) :
    print summary.xlist[idx]
