from buildings_data import BAT_list, SENSORS_list

import matplotlib.pyplot as plt
from matplotlib import path
import numpy as np
from sklearn import neighbors


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
    for bat_num in range(len(BAT_list)):
        bat_path = path.Path(BAT_list[bat_num])
        in_bat = bat_path.contains_points(pts_coord)
        for k in range(0,len(in_bat)):
            if in_bat[k]:
                pts_map[k//n_edge,k%n_edge] = 0
    return(pts_map)


def show_bat(bat_coord, marker='ro-', show=False):
    for i in range(len(BAT_list)):
        the_bat = BAT_list[i]
        the_bat.append(BAT_list[i][0])
        for k in range(0,len(the_bat)-1):
            plt.plot([the_bat[k][0],the_bat[k+1][0]], [the_bat[k][1],the_bat[k+1][1]], marker)
    if show:
        plt.show()


def show_sensors(sens_list, col='gold', show=False):
    for i in range(len(sens_list)):
        sensor = sens_list[i]
        plt.scatter(sensor[0], sensor[1], s=50.0, marker="*", color=col)
        plt.text(sensor[0]+1, sensor[1]+1, "ST"+str(i), fontsize=9)
    if show:
        plt.show()


def show_boundaries(col='black', show=False):
    plt.plot([0,100], [100,100], color=col)
    plt.plot([100,100], [0,100], color=col)
    plt.plot([0,0], [0,100], color=col)
    plt.plot([0,100], [0,0], color=col)
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


### Launch

if __name__ == "__main__":
    graph = make_graph(BAT_list, edge_size=10)
    # show_graph(graph, 10)

    show_bat(BAT_list, marker='b-')
    show_sensors(SENSORS_list)
    show_boundaries()
    plt.axis('equal')
    plt.show()




