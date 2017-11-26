import sys


def anneal(annealer, Ginit = 5., Gfin = 0.01, kT = 0.02, tau = 0.99, n_repeat = 10, verbose = False) :
    Emin = sys.float_info.max
    q0 = []
    q1 = []
    
    for loop in range(0, n_repeat) :
        annealer.randomize_q(0)
        annealer.randomize_q(1)
        G = Ginit
        while Gfin < G :
            annealer.anneal_one_step(G, kT)
            if verbose :
                E - annealer.calculate_E()
                print E
            G = G * tau

        E = annealer.calculate_E()
        if E < Emin :
            q0 = annealer.get_q(0)[0] 
            q1 = annealer.get_q(1)[0]
            Emin = E

    return E, q0, q1
