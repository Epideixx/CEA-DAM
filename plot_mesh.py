import matplotlib.pyplot as plt
from matplotlib import path
import numpy as np
from sklearn import neighbors

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

def get_pts_coord(pts_map,n,e):
    pts_coord = []
    for y in range(n):
        for x in range(n):
            pts_coord.append((x*e,y*e))
    return(pts_coord)

def make_graph(bat_coord, map_size=100.0, edge_size=10):
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

def show_bat(bat_coord, marker='ro-', show=False):
    for i in range(len(Bat_list)):
        the_bat = Bat_list[i]
        the_bat.append(Bat_list[i][0])
        for k in range(0,len(the_bat)-1):
            plt.plot([the_bat[k][0],the_bat[k+1][0]], [the_bat[k][1],the_bat[k+1][1]], marker)
    if show:
        plt.show()

def show_sensors(sens_list, show=False):
    for i in range(len(sens_list)):
        sensor = sens_list[i]
        plt.scatter(sensor[0], sensor[1], marker="*", color="black")
        plt.text(sensor[0]+1, sensor[1]+1, "ST"+str(i), fontsize=9)
    if show:
        plt.show()

def show_boundaries(xmin=0, xmax=100, ymin=0, ymax=100, col='red', show=False):
    plt.plot([xmin,xmax], [ymin,ymin], color=col)
    plt.plot([xmin,xmax], [ymax,ymax], color=col)
    plt.plot([xmin,xmin], [ymin,ymax], color=col)
    plt.plot([xmax,xmax], [ymin,ymax], color=col)
    if show:
        plt.show()


### Show map & graph

def show_graph(pts_map, e, show=False):
    n_pts = pts_map.shape[0]
    for y in range(0,n_pts):
        for x in range(0,n_pts):
            if pts_map[y,x]:
                if y<(n_pts-1) and pts_map[y+1,x]:
                    plt.plot([x*e,x*e], [y*e,(y+1)*e], 'bo-')
                if x<(n_pts-1) and pts_map[y,x+1]:
                    plt.plot([x*e,(x+1)*e], [y*e,y*e], 'bo-')
                plt.scatter(x*e, y*e, color='blue')
    if show:
        plt.show()

def get_edge_maps(pts_map):
    n_edge = pts_map.shape[0]-1
    h_edges = np.zeros((n_edge,n_edge))
    v_edges = np.zeros((n_edge,n_edge))
    for y in range(0,n_edge):
        for x in range(0,n_edge):
            if not(pts_map[y,x] and pts_map[y,x+1]):
                h_edges[y,x] = 0
            if not(pts_map[y,x] and pts_map[y+1,x]):
                v_edges[y,x] = 0
    return(h_edges,v_edges)


### Ant Colony Optimisation

def init_ants(n, n_pts):
    """ Initialisation (aleatoire ?) des fourmis sur la map """
    pass

def pick_edge(edges_link):
    """ Renvoie l'indice de l'arrete choisie pour la transition """
    pass

def time_step(pos_ants, pts_map): #, he, ve):
    n_ants = len(pos_ants)
    new_pos_ants = np.zeros(n_ants)
    transit_edges = np.zeros(n_ants)
    for ant_num in range(len(pos_ants)):
        pts_dest, edges_link = neighbors(pts_map, pos_ants[ant_num])
        i = pick_edge(edges_link)
        new_pos_ants[ant_num] = pts_dest[i]
        transit_edges[ant_num] = edges_link[i]
    return(new_pos_ants, transit_edges)


def ACO(n_ants, pts_map, edge_size, h_edges, v_edges):
    pos_ants = init_ants(n_ants, pts_map.shape[0])
    t = 3
    n_step = 10
    for k in range(0,n_step):
        for u in range(0,t):
            pos_ants, transit_edges = time_step(pos_ants, pts_map, h_edges, v_edges)





graph = make_graph(Bat_list, edge_size=10)

# show_graph(graph, 10)
show_bat(Bat_list, marker='b-')
show_sensors(Sensors_list)
show_boundaries()
plt.axis('equal')
plt.show()

# print(graph)




