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


### 1 - BASIC FEATURES

def create_fname(X, Y, XYindex, round=4):
    return( "CAS_" + str(XYindex) + "_" + str(X) + "_" + str(Y) )

'''
def eval_solution(XS, YS, id):
    CAS_folder_name = create_fname(XS,YS,id)
    CAS_SB2D(CAS_folder_name, AMR, PAS, M_TNT, TPS_F, XS, YS)
    return(cost_function(CAS_folder_name))
'''

def eval_solution(x,y,id=0):
    """ Toy example """
    x -= 30 ; y -= 50
    f_x = 1 - np.exp(-0.0005*(x*x + y*y)) # / sqrt(2*pi)
    return(f_x)


def get_random_pt(map_size,bat_list):
    """ Creates a random valid point on the map. """
    valid_pt = False
    while not(valid_pt):
        px = map_size*random.random() ; py = map_size*random.random()
        valid_pt = not(is_inside_bat((px,py),bat_list))
    return( (px,py) )


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


def smart_insert(l,elt,id):
    """ l is a sorted list of (a,b,f(a,b)) with the first values having little values of f(.,.).
        Inserts from the end the point (elt[0],elt[1],f(elt[0],elt[1])). """
    elt_cost = eval_solution(elt[0],elt[1],id)
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


def create_children(d, rr, mut_std, pt_ID, map_size, bat_list):
    """ Creates a number a children equals to r% the initial population d. Keeps the order in list d. """
    n_children = int(rr*len(d))
    index = [j for j in range(len(d))]
    random.shuffle(index)
    for k in range(n_children):
        i1 = index.pop() ; i2 = index.pop()
        d = smart_insert(d, get_a_child(d[i1], d[i2], mut_std, map_size, bat_list), pt_ID)
        pt_ID += 1
    return(d, pt_ID)


### 3 - DEATH FEATURES

def kill_some_pts(pts, dr, di):
    """ Kills the worst dr% points, each one of them having a last di% chance to survive.
        And each of the non-selected (1-dr)% points have di% chance to die too. """
    survival_list = [True for k in range(len(pts))]
    for k in range(1,int(len(pts)*dr)+1): # worst points that are deemed to die at the beginning
        survival_list[len(pts)-k] = False
    new_pts = []
    for k in range(0,len(pts)):
        if random.random()<=di: # a chance to survive or a risk to die
            survival_list[k] = not(survival_list[k])
        if survival_list[k]: # the survivors
            new_pts.append(pts[k])
    return(new_pts)


### 4 - INITIALIZATION

def init_gen_algo(n, map_size, bat_list):
    """ Gets n random points to initialize the genetic algo. """
    data = [] ; pt_ID = 0
    for k in range(n):
        (ax,ay) = get_random_pt(map_size,bat_list)
        data.append( (ax, ay, eval_solution(ax,ay,pt_ID)) )
        pt_ID += 1
    data = sorted(data, key = lambda u : u[2])
    return(data,pt_ID)


### 5 - GENETIC ALGORITHM

def genetic_algo(map_size, bat_list, n_pts, n_gen=10, death_rate=0.1, death_immun=0.05, repro_rate=0.2, mut_std=1.0, show=False, savefile=None):
    """ Runs an optimization process on a map with buildings using a genetic method. """

    data, pt_ID = init_gen_algo(n_pts, map_size, bat_list) # ; print("DATA",data)
    data_history = [list(data)]
	
    for gen_k in range(n_gen):
        if show:
            print( "\n### Generation {0}".format(gen_k+1) )

        # REPRODUCTION
        data, pt_ID = create_children(data, repro_rate, mut_std, pt_ID, map_size, bat_list) # data is still sorted regarding the 3rd component of each elt
        if show:
            print( "> REPRODUCTION : {0} individuals - BEST = {1}".format(len(data),data[0]) )

        # DEATH
        data = kill_some_pts(data, death_rate, death_immun) # data is still sorted regarding the 3rd component of each elt
        if show:
            print( "> DEATH : {0} individuals - BEST = {1}".format(len(data),data[0]) )
        
        data_history.append(list(data))
    
    # if show:
      #  plot_3D(eval_solution, dx=30, dy=50, z_level=0, points_series=[data])
    
    if savefile != None:
        write_in_txt(data_history, fname=savefile)

    return(data)


### 6 - LAUNCH !

if 0 and __name__ == "__main__":

    final_points = genetic_algo(map_size = 100,
                                bat_list = BAT_list,
                                n_pts = 50,
                                n_gen = 20,
                                death_rate = 0.1,
                                death_immun = 0.05,
                                repro_rate = 0.2,
                                mut_std = 1.0,
                                show = False,
                                savefile = "test_gen_1.txt")

plot_3D(eval_solution, "test_gen_1.txt", dx=30, dy=50, z_level=0, bat_list=BAT_list)
