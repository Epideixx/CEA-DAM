from SouffleBati2D import *
from costfunction import cost_function
from neighbors import get_neighbours
import os


### PAS = 0.4 => environ 1min de simulation
# CAS_SB2D(NOM, AMR, PAS, M_TNT, TPS_F, XC, YC)
# launch_run("")

base_folder_name = "results"

def update_results_file(filename,to_write):
    with open(filename,'a') as f:
        f.write(to_write)


base_signal_folder = "signal_reel"

def create_fname(XY, XYindex):
    return(str(XYindex) + "_" + str(XY[0]) + "_" + str(XY[1]) )


def eval_solution(XS, YS, CAS_folder_name):
    CAS_SB2D(CAS_folder_name, AMR, PAS, M_TNT, TPS_F, XS, YS)
    launch_run("")
    return(cost_function(base_signal_folder,CAS_folder_name))


def greedy_search(X0, Y0, grid_step):
    XYbest = (X0,Y0)
    XYindex = 0
    fname = create_fname(XYbest, XYindex)
    best_cost = eval_solution(XYbest[0], XYbest[1], fname)


    results_file_name = "../log_" + str(X0) + "_" + str(Y0)


    update_results_file(results_file_name,"----------------------SIMULATION RESULTS----------------------")
    update_results_file(results_file_name,"\nStarting point (x,y) : (" + str(X0) + "," + str(Y0) + ")")



    # Get neighbours
    
    neighborhood = get_neighbours(XYbest[0], XYbest[1], grid_step)

    k = 0 ; kmax = 10
    is_new_sol = True
    # Search loop
    while (k<kmax and is_new_sol) :
        update_results_file(results_file_name,"\n----------------------iteration #" + str(k) + "----------------------")
        update_results_file(results_file_name,"\nCurrent point: (" + str(XYbest[0]) + "," + str(XYbest[1]) + ")")
        update_results_file(results_file_name,"\nCurrent cost: " + str(best_cost))
        
        XYbest_neighbor = neighborhood.pop()
        XYindex += 1
        fname = create_fname(XYbest_neighbor,XYindex)
        best_neighbor_cost = eval_solution(XYbest_neighbor[0], XYbest_neighbor[1], fname)

	# Local search loop
        for XY in neighborhood:
            XYindex += 1
            fname = create_fname(XY,XYindex)
            new_cost = eval_solution(XY[0], XY[1], fname)
            
            update_results_file(results_file_name,"\nCurrent neighbor: (" + str(XY[0]) + "," + str(XY[1]) + ")")
            update_results_file(results_file_name,"\nCost: " + str(new_cost))
	    
            if new_cost < best_neighbor_cost:
                XYbest_neighbor = XY
                best_neighbor_cost = new_cost
                update_results_file(results_file_name,"\nNeighbor is current best neighbor")

	    
            if best_neighbor_cost < best_cost : 
                XYbest = XYbest_neighbor
                best_cost = best_neighbor_cost
                neighborhood = get_neighbours(XYbest[0], XYbest[1], grid_step)
                update_results_file(results_file_name,"\nNeighbor is better than current solution: changing current best")



if __name__ == "__main__":
    X0 = 45
    Y0 = 70
    greedy_search(X0, Y0, grid_step=1)
