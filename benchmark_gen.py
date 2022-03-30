import os
from opti_genetic import *
from plot_gen_data import *
from buildings_data import *


def get_centroid(d,p=None):
    """ Compute the centroid of the first p% pts in d. """
    cx = 0 ; cy = 0
    if p!=None:
        nc = int(p*len(d))
    else:
        nc = len(d)
    for k in range(0,nc):
        cx += d[k][0] ; cy += d[k][1]
    cx = float(cx)/nc ; cy = float(cy)/nc
    return( (cx,cy) )


# Parameters set for genetic algorithm benchmark

PSET = {"map_size":100, "bat_list":BAT_list, "corner_pts":True, "n_pts":40, "n_gen":20,
        "death_rate":0.25, "death_flip":0.05, "death_immun":0.1, "repro_rate":0.25, "mut_std":3.0}


# Benchmark

def benchmark(n_exec = 20, pset = PSET, src = None):
    """ Benchmark de test de l'algo genetique. """
    benchmark_folder = "BENCHMARK_" + str(n_exec) + "/"
    os.mkdir("gen_data/" + benchmark_folder)

    best_pts = np.zeros((2,n_exec))
    centroid_pts = np.zeros((2,n_exec))
    for simu_gen in range(0,n_exec):
        file_text = benchmark_folder + "benchmark_" + str(simu_gen) + "-" + str(n_exec)
        
        final_pts = genetic_algo(pset["map_size"], pset["bat_list"], pset["n_pts"], pset["corner_pts"], pset["n_gen"],
                                 pset["death_rate"], pset["death_flip"], pset["death_immun"], pset["repro_rate"], pset["mut_std"], 
                                 show=False, savemode="auto", savefile=file_text, resumefile=None) # Simulation
        
        centroid = get_centroid(final_pts,p=0.5)

        centroid_pts[0,simu_gen] = centroid[0]
        centroid_pts[1,simu_gen] = centroid[1]

        best_pts[0,simu_gen] = final_pts[0][0]
        best_pts[1,simu_gen] = final_pts[0][1]

        print("SIMU {0}/{1} : centroid = ({2},{3}) & best = ({4},{5})\n".format(simu_gen,n_exec,centroid[0],centroid[1],final_pts[0][0],final_pts[0][1]))
    
    # Plot centroids and best pts
    plt.figure(figsize=(10, 7))
    show_bat(BAT_list)
    if src != None:
        plt.scatter(src[0], src[1], s=50.0, marker="s", color="black", label="SRC ({0},{1})".format(src[0],src[1]))
    show_boundaries()
    plt.scatter(centroid_pts[0,:], centroid_pts[1,:], color='b', label="CENTROIDS")
    plt.scatter(best_pts[0,:], best_pts[1,:], color='green', marker='^', label="BEST PTS")
    plt.axis('equal') ; plt.legend(loc='upper left', fontsize=8)
    plt.show()

    return(centroid_pts,best_pts)


# Launch benchmark

centroid_pts, best_pts = benchmark(n_exec=50, src=(22,20))