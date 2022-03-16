# from logging.handlers import WatchedFileHandler
#from SouffleBati2D import *
# from costfunction import cost_function
from is_inside import is_inside as is_inside_poly
import random
import numpy as np
from math import sqrt, exp, pi
import matplotlib.pyplot as plt


### PAS = 0.4 => environ 1min de simulation
# CAS_SB2D(NOM, AMR, PAS, M_TNT, TPS_F, XC, YC)
# launch_run("")


### 0 - MAP FEATURES

Bat1 = [(10.08,10.223),
        (30.08,10.223),
        (30.08,15.223),
        (17.08,15.223),
        (17.08,25.223),
        (10.08,25.223)]

Bat2 = [(50.08,12.223),
        (65.08,12.223),
        (65.08,27.223),
        (50.08,27.223)]

Bat3 = [(80.08,32.223),
        (87.08,32.223),
        (87.08,57.223),
        (80.08,57.223)]

Bat4 = [(12.08,78.223),
        (42.08,78.223),
        (42.08,92.223),
        (12.08,92.223)]

Bat5 = [(75,70),
        (85,80),
        (90,75),
        (96,81),
        (0.5*167,0.5*187),
        (0.5*135,0.5*155)]

Bat6 = [(20.08,40.223),
        (40.08,40.223),
        (40.08,45.223),
        (25.08,45.223),
        (25.08,55.223),
        (35.08,55.223),
        (35.08,70.223),
        (20.08,70.223)]

Bat_list = [Bat1,Bat2,Bat3,Bat4,Bat5,Bat6]

Sensors_list = [(12.82,33.83),
                (42.21,95.27),
                (85.95,65.66),
                (92.13,53.32),
                (61.87,31.51)]


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


def show_3D(f, points_series, dx, dy, z_level, bat_list=Bat_list, colors=['red','blue','green','gray']):
    ## Compute values
    X, Y = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
    Z = f(X,Y)

    ## Plot 2D shape
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ## Plot bat
    if bat_list != None:
        for i in range(len(Bat_list)):
            the_bat = Bat_list[i] ; the_bat.append(Bat_list[i][0])
            for k in range(0,len(the_bat)-1):
                ax.plot([the_bat[k][0]-dx,the_bat[k+1][0]-dx], [the_bat[k][1]-dy,the_bat[k+1][1]-dy], z_level, 'k-')

    ## Plot points
    for snum in range(0,len(points_series)):
        series = points_series[snum]
        for (x,y,z) in series:
            ax.scatter(x-dx,y-dy, z_level, color=colors[snum])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # ax.view_init(60, 35)
    # fig


def is_inside_bat(x, bat_list):
    """ Checks if point x is inside a bat defined in bat_list or not. """
    is_inside_bat = False ; i = 0
    while not(is_inside_bat) and i<len(bat_list):
        is_inside_bat = is_inside_poly(x, bat_list[i])
        i += 1
    return(is_inside_bat)


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

def genetic_algo(map_size, bat_list, n_pts, n_gen=10, death_rate=0.1, death_immun=0.05, repro_rate=0.2, mut_std=1.0, show=False):
    """ Runs an optimization process on a map with buildings using a genetic method. """

    data_init, pt_ID = init_gen_algo(n_pts, map_size, bat_list) # ; print("DATA",data)
    data = data_init
	
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

    if show:
        show_3D(eval_solution, dx=30, dy=50, z_level=0, points_series=[data_init,data])

    return(data)


### 6 - WRITE/READ DATA

def write_in_txt(d,fname):
    f = open("gen_data/" + fname, "w")
    for (ax,ay,cost_a) in d:
        f.write("{0};{1};{2}\n".format(ax,ay,cost_a))
    f.close()

def read_in_txt(fname):
    d = []
    f = open("gen_data/" + fname, "r")
    txt_lines = f.read().split("\n")
    for line in txt_lines:
        if len(line)>0:
            elts = line.split(";")
            d.append( (float(elts[0]),float(elts[1]),float(elts[2])) )
        else:
            print("EMPTY LINE FOUND...")
    f.close()
    return(d)


### 6 - LAUNCH !

if 1 and __name__ == "__main__":

    final_points = genetic_algo(map_size = 100,
                                bat_list = Bat_list,
                                n_pts = 50,
                                n_gen = 20,
                                death_rate = 0.1,
                                death_immun = 0.05,
                                repro_rate = 0.2,
                                mut_std = 1.0,
                                show = True)

    write_in_txt(final_points, fname = "test_gen_1.txt")
