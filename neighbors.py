#from SouffleBati2D import *
from is_inside import *
import numpy as np
import matplotlib.pyplot as plt

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

Bat_list = [[tuple([5*x for x in i]) for i in Bat] for Bat in [Bat1,Bat2,Bat3,Bat4,Bat5,Bat6]]
  
def get_pts_coord(pts_map,n,e):
    pts_coord = []
    for y in range(n):
        for x in range(n):
            pts_coord.append((x*e,y*e))
    return(pts_coord)

def make_graph(bat_coord, map_size=100.0, edge_size=10.0):
    """ Map de map_size x map_size avec un graph grille d'arretes de taille edge_size """
    n_edge = int(map_size//edge_size)
    pts_map = np.ones((n_edge,n_edge))
    pts_coord = get_pts_coord(pts_map, n=n_edge, e=edge_size)
    for bat in Bat_list:
        for e in pts_coord :
            if is_inside(e, bat) :
                pts_map[int(e[0])//edge_size,int(e[1])//edge_size] = 0
    return(np.rot90(pts_map, k=1, axes=(0, 1)))

def get_neighbors(pts_map, x0, y0):

    neighbors=[]
    num_neighbor = 1
    index = [x0, y0]

    left = max(0,index[0]-num_neighbor)
    right = max(0,index[0]+num_neighbor)

    bottom = max(0,index[1]-num_neighbor)
    top = max(0,index[1]+num_neighbor)

    neighbors = [(left, top), (x0, top), (right, top), (left, y0), (right, y0), (left, bottom), (x0, bottom), (right, bottom)]

    for e in neighbors:
        if e == (x0,y0):
            neighbors.remove(e)
        if pts_map(e) == 0:
            neighbors.remove(e)
    return neighbors

G = make_graph(Bat_list, 500, 1)

plt.imshow(G,interpolation='nearest', cmap = 'gray')
plt.show()