from buildings_data import BAT_list, SENSORS_list

from plot_mesh import show_bat,show_sensors,show_boundaries
from read_write_gen_data import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

COLORS = ['red','blue','green','gray','yellow']


def tab_eval(f, tabX, tabY):
    """ Evaluate f on the mesh defined by tabX and tabY. """
    (nr,nc) = tabX.shape
    tabf = np.zeros((nr,nc))
    for row in range(0,nc):
        for col in range(0,nc):
            tabf[row,col] = f(tabX[row,col],tabY[row,col])
    return(tabf)


def plot_2D(file_name, modulo=1, gen_num=None, bat_list=None, sens_list=None, src=None):
    """ Plot several generations on a map with the buildings, sensors and boundaries. """
    dhist = read_in_txt(file_name) # fetch the data
    
    plt.figure(figsize=(10, 7))
    if bat_list != None:
        show_bat(bat_list)
    if sens_list != None:
        show_sensors(sens_list)
    if src != None:
        plt.scatter(src[0], src[1], s=50.0, marker="s", color="black", label="SRC ({0},{1})".format(src[0],src[1]))
    show_boundaries()
    
    if gen_num == None: # modulo gen plot
        for g in range(0,len(dhist)):
            if (g+1)%modulo == 0:
                gen_data = dhist[g]
                plt.scatter(gen_data[0,1:], gen_data[1,1:], color=cm.winter(g/float(len(dhist))), label="GEN {0} | #{1}".format(g+1,gen_data.shape[1]))
                plt.scatter(gen_data[0,0], gen_data[1,0], color=cm.winter(g/float(len(dhist))), marker='^', )
    else: # single gen plot
        gen_data = dhist[gen_num-1]
        plt.scatter(gen_data[0,1:], gen_data[1,1:], color=cm.winter(gen_num/float(len(dhist))), label="GEN {0} | #{1}".format(gen_num,gen_data.shape[1]))
        plt.scatter(gen_data[0,0], gen_data[1,0], color=cm.winter(gen_num/float(len(dhist))), marker='^', )
    

    plt.axis('equal') ; plt.legend(loc='upper left', fontsize=8)
    plt.show()



def plot_3D(f, file_name, bat_list, map_level=0):
    """ Plot in 3D the objective function, the buildings, boundaries and the first and final generations. """
    ## Fetch the data
    if file_name != None:
        dhist = read_in_txt(file_name)
    
    ## Compute mesh for 3D shape
    X, Y = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
    Z = tab_eval(f, np.array(X), np.array(Y))
    
    ## Plot 3D shape
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary') # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.set_xlim3d(0, 100) ; ax.set_ylim3d(0, 100) ; ax.set_zlim3d(0, 2)

    if file_name != None:
        ## Plot buildings & boundaries
        if bat_list != None:
            for i in range(len(bat_list)):
                the_bat = bat_list[i] ; the_bat.append(bat_list[i][0])
                for k in range(0,len(the_bat)-1):
                    ax.plot([the_bat[k][0],the_bat[k+1][0]], [the_bat[k][1],the_bat[k+1][1]], map_level, 'k-')
        ax.plot([0,100], [100,100], [map_level,map_level], color="black") ; ax.plot([100,100], [0,100], [map_level,map_level], color="black")
        ax.plot([0,0], [0,100], [map_level,map_level], color="black") ; ax.plot([0,100], [0,0], [map_level,map_level], color="black")

        ## Plot points
        for g in [0,len(dhist)-1]: # plot only the first and last generations
            gen_data = dhist[g]
            ax.scatter(gen_data[0,1:], gen_data[1,1:], map_level, c=COLORS[g%len(COLORS)], label="GEN {0} | #{1}".format(g,gen_data.shape[1]))
            ax.scatter([gen_data[0,0]], [gen_data[1,0]], map_level, c=COLORS[g%len(COLORS)], marker='^')

    ax.set_xlabel('x') ; ax.set_ylabel('y') ; ax.set_zlabel('z')
    plt.show()


if __name__ == "__main__":
    plot_2D("BENCHMARK_10/benchmark_5-10",
            modulo = None,
            gen_num = 19,
            bat_list = BAT_list, 
            sens_list = SENSORS_list, 
            src = (22,20))


