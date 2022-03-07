from mySouffleBati2D import *
from mycostfunction import cost_function
from myneighborhood import get_neighbours


### PAS = 0.4 => environ 1min de simulation
# CAS_SB2D(NOM, AMR, PAS, M_TNT, TPS_F, XC, YC)
# launch_run("")


def create_fname(XY, XYindex):
    return( "CAS_" + str(XYindex) + "_" + str(XY[0]) + "_" + str(XY[1]) )


def eval_solution(XS, YS, CAS_folder_name, PAS_simul):
    CAS_SB2D(CAS_folder_name, AMR, PAS_simul, M_TNT, TPS_F, XS, YS)
    return(cos_function(CAS_folder_name))


def greedy_search(X0, Y0, PAS_simul, grid_step):
    """ Local greedy search algorithm """

    # Init
    XYbest = (X0,Y0)
    XYindex = 0
    fname = create_fname(XYbest, XYindex)
    best_cost = eval_solution(XYbest[0], XYbest[1], fname, PAS_simul)

    # Get neighbours
    neighborhood = get_neighbours(XYbest[0], XY_best[1], grid_step)

    k = 0 ; kmax = 10
    is_new_sol = True
    # Search loop
    while (k<kmax and is_new_sol):

	# Init local search
	XY_best_neighbor = pop(neighborhood)
	XYindex += 1
	fname = create_fname(XY,XYindex)
	best_neighbor_cost = eval_solution(XY[0], XY[1], fname, PAS_simul)

	# Local search loop
	for XY in neighborhood:
	    XYindex += 1
	    fname = create_fname(XY,XYindex)
	    new_cost = eval_solution(XY[0], XY[1], fname, PAS_simul)
	    if new_cost < best_neighbor_cost:
		XY_best_neighbor = XY
                best_neighbor_cost = new_cost

	if best_neighbor_cost < best_cost: # Better neighbor found
	    XYbest = XY_best_neighbor
	    best_cost = best_neighbor_cost
	    neighborhood = get_neighbours(XYbest[0], XYbest[1], grid_step)
	else: # End of 'Search loop'
	    is_new_sol = False

	k+=1

    print(XYbest, best_cost)


if __name__ == "__main__":
    greedy_search(X0, Y0, PAS_simul=0.4, grid_step=???)
