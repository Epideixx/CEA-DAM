from SouffleBati2D.py import *
from sklearn import neighbors
from matplotlib import path

Bat_list = [Bat1,Bat2,Bat3,Bat4,Bat5,Bat6]
  
def get_pts_coord(pts_map,n,e):
    pts_coord = []
    for y in range(n):
        for x in range(n):
            pts_coord.append((x*e,y*e))
    return(pts_coord)

def make_graph(bat_coord, map_size=100.0, edge_size):
    """ Map de map_size x map_size avec un graph grille d'arretes de taille edge_size """
    n_edge = int(map_size//edge_size)
    pts_map = np.ones((n_edge,n_edge))
    pts_coord = get_pts_coord(pts_map, n=n_edge, e=edge_size)
    for bat_num in range(len(Bat_list)):
        bat_path = path.Path(Bat_list[bat_num])
        in_bat = bat_path.contains_points(pts_coord)
        for k in range(0,len(in_bat)):
            if in_bat[k]:
                pts_map[k//n_edge,k%n_edge] = 0
    return(pts_map)

def get_neighbors(x0, y0, edge_size):
    pts_map = make_graph(Bat_list, edge_size)

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
