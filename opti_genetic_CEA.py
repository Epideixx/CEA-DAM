# from logging.handlers import WatchedFileHandler
#from SouffleBati2D import *
# from costfunction import cost_function

from buildings_data import BAT_list, SENSORS_list
from is_inside import is_inside_bat
from plot_gen_data import *
from read_write_gen_data import *

import random
import numpy as np
from math import sqrt, exp, pi
# import matplotlib.pyplot as plt
import os
import shutil
from concurrent.futures import ThreadPoolExecutor


# ---- Module for CEA ----
from Simulation2D import main


### 1 - BASIC FEATURES

def create_fname(X, Y, XYindex, round=4):
    return( "CAS_" + str(XYindex) + "_" + str(X) + "_" + str(Y) )

'''
def eval_solution(XS, YS, id):
    CAS_folder_name = create_fname(XS,YS,id)
    CAS_SB2D(CAS_folder_name, AMR, PAS, M_TNT, TPS_F, XS, YS)
    return(cost_function(CAS_folder_name))
'''

PARAM = np.array([ [50.0,  50.0, 0.001,  -1.0,  1.0],
                   [22.0,  20.0,  0.01,  3.0,  1.0],
                   [88.0,  78.0, 0.003,  2.0,  10.0],
                   [90.0,  20.0,  0.01,  1.0,  -1.0],
                   [30.0,  75.0, 0.008,  1.2,  0.0] ]) # [px,py,in_scale,out_scale,out_offset]


def eval_solution_test(x, y, id=None):
    """ Evaluate quality of a source point (x,y). id is used to get a unique ID for the files
        from the CEA simulation. """
    fx = 0.0
    for i in range(PARAM.shape[0]):
        fx += PARAM[i,4] - (PARAM[i,3] * np.exp( -PARAM[i,2] * ( (x-PARAM[i,0])*(x-PARAM[i,0]) + (y-PARAM[i,1])*(y-PARAM[i,1]) )))
    return(fx)

def eval_solution(x, y, id = None, step_size = 0.4, repertory = "Genetic"):
    """ Evaluate quality of a source point (x,y). id is used to get a unique ID for the files
        from the CEA simulation with a given step size"""
    
    # We launch the simulation
    filename = "Simulation_"+ str(x) + "_" + str(y) + "__" + str(step_size)
    main(filename = filename, XC = x, YC = y, pas = step_size, M_TNT = 10) #cf. CEA

    # We keep only the TE file that we put in the repertory
    if not os.path.exists(repertory):
        os.makedirs(repertory)

    src = os.path.join(filename, "POST1D", "TE")
    dest = os.path.join(repertory, filename)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    shutil.rmtree(filename)

    # We evaluate the point with the data
    # TO UNCOMMENT
    """
    eval = cost_function(dest)

    return eval
    """


def get_random_pt(map_size,bat_list):
    """ Creates a random valid point on the map. """
    valid_pt = False
    while not(valid_pt):
        px = map_size*random.random() ; py = map_size*random.random()
        valid_pt = not(is_inside_bat((px,py),bat_list))
    return( (px,py) )

def choose_step(n_population):
    """ Returns a step size depending of the size of the population"""
    if n_population >= 100 :
        step_size = 1
    elif n_population >=75 :
        step_size = 0.8
    elif n_population >= 50 :
        step_size = 0.6
    elif n_population >= 25 :
        step_size = 0.5
    elif n_population >= 15 :
        step_size = 0.4
    else :
        step_size = 0.3
    
    return step_size


### 2 - REPRODUCTION FEATURES

def dashed_line(a,b,n):
    """ Creates n points regularly set on the line between point a and point b. """
    u = ( (b[0]-a[0])/(n+1) , (b[1]-a[1])/(n+1) ) # step vector from a to b
    dline = [ ( a[0]+k*u[0] , a[1]+k*u[1] ) for k in range(1,n+1) ]
    return(dline)


def mutate_pt(a, std, bat_list, lim = 20):
    """ Considering a valid point a (i.e. not in a building), try to mutate a into a valid point using 
        a gaussian (0,std) noise in both X and Y directions. 
        If the mutant isn't valid, retry at most lim times. """
    valid_mut = False ; m = 0
    while not(valid_mut) and (m<lim):
        (ex,ey) = np.random.normal(0,std,2)
        new_a = (a[0]+ex , a[1]+ey)
        valid_mut = not(is_inside_bat(new_a, bat_list)) ; m += 1
    if not(valid_mut): # don't mutate if it's too hard
        new_a = a
    return(new_a)


def smart_insert(l,elt,id, step_size = 0.4, repertory = "Genetic"):
    """ l is a sorted list of (a,b,f(a,b)) with the first values having little values of f(.,.).
        Inserts from the end the point (elt[0],elt[1],f(elt[0],elt[1])). """
    elt_cost = eval_solution(elt[0],elt[1],id, step_size= step_size, repertory = repertory)
    l.append( (elt[0],elt[1],elt_cost) )
    i = len(l)-1 ; switch = True
    while switch and (i>0):
        if l[i-1][2]>l[i][2]: # compare the 3rd component of each element
            buff = l[i-1]
            l[i-1] = l[i]
            l[i] = buff
            i -= 1
        else:
            switch = False
    return(l)

def smart_insert_multithread(elt,id, step_size = 0.4, repertory = "Genetic"):
    elt_cost = eval_solution(elt[0],elt[1],id, step_size= step_size, repertory = repertory)
    return (elt[0],elt[1],elt_cost)


def get_a_child(a, b, std, map_size, bat_list):
    """ Creates a child from two parents a and b. The default child is the mean of a and b + mutation.
        Otherwise, we search a valid child from the points of the dashed line between a and b + mutation.
        If there is no valid point among them, we create a random valid point on the map. """
    mean_pt = ( (a[0]+b[0])/2 , (a[1]+b[1])/2 ) # default point
    ghost_pt = is_inside_bat(mean_pt, bat_list)

    if ghost_pt: # the mean point is inside a bat
        line_pts = dashed_line(a,b,n=20) # create candidates for child_pt
        random.shuffle(line_pts)
        valid_pt = False
        while not(valid_pt) and len(line_pts)>0:
            x = line_pts.pop()
            if not(is_inside_bat(x, bat_list)): # check if the choosen candidate is valid
                valid_pt = True ; child_pt = x
        if not(valid_pt):
            child_pt = get_random_pt(map_size,bat_list) # get a random valid point on the map
    else: # the mean point is valid
        child_pt = mean_pt

    return(mutate_pt(child_pt, std, bat_list)) # return a mutant from the valid child_pt


def create_children(d, rr, mut_std, pt_ID, map_size, bat_list, step_size=0.4, repertory="Genetic", multithreading = False):
    """ Creates a number a children equals to r% the initial population d. Keeps the order in list d. """
    n_children = int(rr*len(d))
    index = [j for j in range(len(d))]
    random.shuffle(index)
    pt_ID_init = pt_ID
    if multithreading:
        in_simu = {}
        executor = ThreadPoolExecutor(max_workers=4)
        for k in range(n_children):
            i1 = index.pop() ; i2 = index.pop()
            in_simu[pt_ID] = executor.submit(smart_insert_multithread, d, get_a_child(d[i1], d[i2], mut_std, map_size, bat_list), pt_ID,step_size,repertory)
            pt_ID += 1
        pt_ID = pt_ID_init
        d = []
        for k in range(n_children):
            d.append(in_simu[pt_ID].result())
            pt_ID += 1
        d = sorted(d, key = lambda x : x[2])
    else :
        for k in range(n_children):
            i1 = index.pop() ; i2 = index.pop()
            d = smart_insert(d, get_a_child(d[i1], d[i2], mut_std, map_size, bat_list), pt_ID, step_size=step_size, repertory=repertory, multithreading=multithreading)
            pt_ID += 1
    return(d, pt_ID, n_children)


### 3 - DEATH FEATURES

def kill_some_pts(pts, dr, df, di):
    """ Kills the worst dr% points, each one of them having a last di% chance to survive.
        And each of the non-selected (1-dr)% points have ds% chance to die too. 
        However the di% best points always survive. """
    survival_list = [True for k in range(len(pts))]
    for k in range(1,int(len(pts)*dr)+1): # worst points that are deemed to die at the beginning
        survival_list[len(pts)-k] = False
    nb_best = int(di*len(pts))
    new_pts = [pts[i] for i in range(0,nb_best)] # the best di% always survive
    for k in range(nb_best,len(pts)): # for all pts except the best di%
        if random.random()<=df: # a chance to survive or a risk to die
            survival_list[k] = not(survival_list[k])
        if survival_list[k]: # the survivors
            new_pts.append(pts[k])
    return(new_pts)


### 4 - INITIALIZATION

def init_gen_algo(n, map_size, bat_list, corner_pts=False, step_size = 0.4, repertory = "Genetic", multithreading = False):
    """ Gets n random points to initialize the genetic algo. """
    data = [] ; pt_ID = 0

    if corner_pts and (n<4):
        raise Exception("ERROR : n={0} not sufficient when considering 8 points on boundaries".format(n))

    if corner_pts:
        if multithreading:
            in_simu = {}
            executor = ThreadPoolExecutor(max_workers=4)
            for (ax,ay) in [(0.0,0.0),(0.0,100.0),(100.0,0.0),(100.0,100.0)]: # corner points
                in_simu[pt_ID] = executor.submit(eval_solution, ax,ay,pt_ID, step_size, repertory)
                pt_ID += 1
            pt_ID = 0
            for (ax,ay) in [(0.0,0.0),(0.0,100.0),(100.0,0.0),(100.0,100.0)]:
                data.append((ax, ay, in_simu[pt_ID].result())) ; pt_ID += 1
        
        else : 
            for (ax,ay) in [(0.0,0.0),(0.0,100.0),(100.0,0.0),(100.0,100.0)]: # corner points
                data.append((ax, ay, eval_solution(ax,ay,pt_ID, step_size, repertory))) ; pt_ID += 1
                pt_ID += 1
    else: # corner_pts = False
        n += 4


    if multithreading:
        pt_init = pt_ID
        for k in range(n-4):
            in_simu = {}
            executor = ThreadPoolExecutor()
            (ax,ay) = get_random_pt(map_size,bat_list)
            in_simu[pt_ID] = executor.submit(eval_solution, ax,ay,pt_ID, step_size, repertory)
            pt_ID += 1
        pt_ID = pt_init
        for k in range(n-4):
            data.append( (ax, ay, in_simu[pt_ID].result()) ) ; pt_ID += 1

    else :
        for k in range(n-4):
            (ax,ay) = get_random_pt(map_size,bat_list)
            data.append( (ax, ay, eval_solution(ax,ay,pt_ID, step_size=step_size, repertory=repertory)) ) ; pt_ID += 1

    data = sorted(data, key = lambda u : u[2])
    return(data,pt_ID)


### 5 - GENETIC ALGORITHM

def genetic_algo(map_size, bat_list, n_pts, corner_pts=False, n_gen=10, death_rate=0.1, death_flip=0.05, death_immun=0.1, repro_rate=0.2, mut_std=1.0, show=False, savemode="auto", savefile=None, resumefile=None, repertory = "Genetic", multithreading = False):
    """ Runs an optimization process on a map with buildings using a genetic method. 
        - corner_pts : <bool> (add one initial pt at each corner of the map = 4 pts)
        - show : <bool> (show the execution of the algo with prints in the shell)
        - savemode : "auto" (save gen n before computing gen n+1) | "end" (save all gen at the end) 
        - savefile : <str> (name of the save file) 
        - resumefile : <str> (run gen algo from data in the file resume)
        - repertory : <str> (name of the foder to put the simulated data
    """

    if resumefile == None:
        step_size = choose_step(n_pts)
        data, pt_ID = init_gen_algo(n_pts, map_size, bat_list, corner_pts, step_size=step_size, repertory=repertory, multithreading = multithreading) # ; print("DATA",data)
        data_history = [list(data)]
    else:
        data_history = read_in_txt(resumefile) # fetch gen data from resumefile
        data = []
        for k in range(0,data_history[-1].shape[1]): # transform last gen from numpy to triplet list
            data.append( (data_history[-1][0,k], data_history[-1][1,k], data_history[-1][2,k]) )
        restart_gen = len(data_history)+1
        savemode = "auto" ; savefile = resumefile ; pt_ID = 0

    gen_k = 0 ; NB_SIM = len(data)
    n_population = NB_SIM
    while gen_k < n_gen:
        if (resumefile != None) and (gen_k == 0): # restart from a given gen and parameters for the simulation
            gen_k = restart_gen

        if show:
            print( "\n### Generation {0}".format(gen_k+1) )

        # Choose step_size for the simulation
        step_size = choose_step(n_population)

        # REPRODUCTION
        data, pt_ID, delta_SIM = create_children(data, repro_rate, mut_std, pt_ID, map_size, bat_list, step_size=step_size, repertory=repertory, multithreading = multithreading) # data is still sorted regarding the 3rd component of each elt
        NB_SIM += delta_SIM
        if show:
            print( "> REPRODUCTION : {0} individuals - BEST = {1}".format(len(data),data[0]) )

        # DEATH
        data = kill_some_pts(data, death_rate, death_flip, death_immun) # data is still sorted regarding the 3rd component of each elt
        n_population = len(data)
        if n_population == 0:
            raise Exception("> ALL DEAD...")
        if show:
            print( "> DEATH : {0} individuals - BEST = {1}".format(len(data),data[0]) )
            print("[nb sim = {0}]".format(delta_SIM))
        
        data_history.append(list(data))
        if savemode == "auto":
            append_in_txt(data,gen_k,fname=savefile)
        
        if n_population == 1: # only one individual left
            print("\n> STOP ALGO : Only 1 individual left in GENERATION {0}.".format(gen_k+1))
            gen_k = n_gen
        gen_k += 1
    
    if savemode == "end":
        write_in_txt(data_history, fname=savefile)

    if show:
        print("\n=> NB SIMULATIONS = {0}".format(NB_SIM))

    return(data)


### 7 - LAUNCH !

if 0 and __name__ == "__main__":

    final_points = genetic_algo(map_size = 100,
                                bat_list = BAT_list,
                                n_pts = 40, # 40
                                corner_pts = True,
                                n_gen = 20, # 20
                                death_rate = 0.25, # 25
                                death_flip = 0.05, # 5
                                death_immun = 0.1, # 5
                                repro_rate = 0.25, # 25
                                mut_std = 3.0,
                                show = True,
                                savemode = "auto",
                                savefile = "test_gen.txt",
                                resumefile = None) #"test_gen.txt")

# plot_3D(eval_solution, "test_gen.txt", map_level=8.0, bat_list=BAT_list)