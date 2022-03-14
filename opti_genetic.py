from SouffleBati2D import *
from costfunction import cost_function
from polygon import is_inside_poly
import random
import numpy as np


### PAS = 0.4 => environ 1min de simulation
# CAS_SB2D(NOM, AMR, PAS, M_TNT, TPS_F, XC, YC)
# launch_run("")

BatiList = []

def create_fname(XY, XYindex):
    return( "CAS_" + str(XYindex) + "_" + str(XY[0]) + "_" + str(XY[1]) )


def eval_solution(XS, YS, CAS_folder_name):
    CAS_SB2D(CAS_folder_name, AMR, PAS, M_TNT, TPS_F, XS, YS)
    return(cost_function(CAS_folder_name))

def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return(lst.pop(idx))

def get_children(d,r):
    index = range(len(d))
    n_children = int(r*len(d))
    new_data = []
    for k in range(n_children):
        i1 = pop_random(index) ; i2 = pop_random(index)
        new_data.append( get_mean(d[i1],d[i2]) )
    return(new_data)

def get_mean(a,b):
    """ Prendre en compte les bâtiments ! """
    return( (a[0]+b[0])/2 , (a[1]+b[1])/2 )

def death_in_action(d, dr, di):
    survival_list = [True for k in range(len(d))]
    for k in range(1,int(len(d)*dr)+1):
        survival_list[len(d)-k] = False
    new_d = []
    for k in range(0,len(d)):
        if random()<=di:
            survival_list[k] = not(survival_list[k])
        if survival_list[k]:
            new_d.append(d[k])
    return(new_d)

def mutate(d, mr, std):
    for k in range(0,len(d)):
        if random()<=mr:
            (ex,ey) = np.random.normal(0,std,2)
            d[k] = (d[k][0]+ex , d[k][1]+ey)
    return(d)

def init_gen_algo(n, map_size=100):
    pts = []
    for k in range(0,n):
        x = map_size*random() ; y = map_size*random()
        while is_inside_poly(x,y,BatiList):
            x = map_size*random() ; y = map_size*random()
        pts.append( (x,y) )
    return(pts)


def genetic_algo(points, n_gen=10, death_rate=0.1, death_immun=0.05, repro_rate=0.2, mut_rate=0.1, mut_std=1.0):
   
    data = [] ; xy_index = 0
    for (x,y) in points:
        data.append( (x,y,eval_solution(x,y,xy_index))) ; xy_index += 1
    data = sorted(data, reverse=True, key = lambda u : u[2])
	
    for gen_k in range(n_gen):

        # DEATH
        data = death_in_action(data, death_rate, death_immun)

        # REPRODUCTION
        new_data = get_children(data,repro_rate)
        for k in range(0,len(new_data)):
            (xk,yk) = new_data[k]
            new_data[k] = (xk, yk, eval_solution(xk,yk,xy_index)) ; xy_index += 1
            
        # MUTATION
        data = mutate(data + new_data, mut_rate, mut_std) ### STD !!!
        
        data = sorted(data, reverse=True, key = lambda u : u[2])
    
    return(data)




if __name__ == "__main__":
    genetic_algo(init_gen_algo(10))
