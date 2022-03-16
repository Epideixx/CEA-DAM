from buildings_data import BAT_list, SENSORS_list

from plot_mesh import show_bat,show_sensors,show_boundaries
from read_write_gen_data import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

COLORS = ['red','blue','green','gray','yellow']


def plot_2D(file_name, dx, dy, modulo=1, bat_list=None, sens_list=None):
    """ Plot several generations on a map with the buildings, sensors and boundaries. """
    dhist = read_in_txt(file_name) # fetch the data
    
    if bat_list != None:
        show_bat(bat_list)
    if sens_list != None:
        show_sensors(sens_list)
    show_boundaries()

    if len(COLORS)<len(dhist):
        print("> Not enough colors !")
    
    for g in range(0,len(dhist)):
        if g%modulo == 0:
            gen_data = dhist[g]
            plt.scatter(gen_data[0,1:], gen_data[1,1:], color=cm.winter(g/float(len(dhist))), label="GEN {0}".format(g))
            plt.scatter(gen_data[0,0], gen_data[1,0], color=cm.winter(g/float(len(dhist))), marker='^', )

    plt.legend()
    plt.show()



def plot_3D(f, file_name, dx, dy, z_level, bat_list):
    """ Plot in 3D the objective function, the buildings, boundaries and the first and final generations. """
    ## Fetch the data
    dhist = read_in_txt(file_name)
    
    ## Compute mesh for 3D shape
    X, Y = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
    Z = f(X,Y)

    ## Plot 3D shape
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ## Plot buildings & boundaries
    if bat_list != None:
        for i in range(len(bat_list)):
            the_bat = bat_list[i] ; the_bat.append(bat_list[i][0])
            for k in range(0,len(the_bat)-1):
                ax.plot([the_bat[k][0]-dx,the_bat[k+1][0]-dx], [the_bat[k][1]-dy,the_bat[k+1][1]-dy], z_level, 'k-')
    ax.plot([0,100], [100,100], color="black") ; ax.plot([100,100], [0,100], color="black")
    ax.plot([0,0], [0,100], color="black") ; ax.plot([0,100], [0,0], color="black")

    ## Plot points
    for g in [0,len(dhist)-1]: # plot only the first and last generations
        gen_data = dhist[g]
        ax.scatter(gen_data[0,1:], gen_data[1,1:], c=COLORS[g%len(COLORS)], label="GEN {0}".format(g))
        ax.scatter([gen_data[0,0]], [gen_data[1,0]], c=COLORS[g%len(COLORS)], marker='^')

    ax.set_xlabel('x') ; ax.set_ylabel('y') ; ax.set_zlabel('z')
    plt.show()


if __name__ == "__main__":
    plot_2D("test_gen_1.txt", dx=30, dy=50, modulo=3, bat_list=BAT_list, sens_list=SENSORS_list)