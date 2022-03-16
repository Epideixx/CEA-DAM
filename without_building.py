# ----------------------------------------------
#          Detect sources without buildings
# ----------------------------------------------

"""
Notes informatives :
- L'équation de l'intersection des cercles est résolu numériquement et non analytiquement
- Il y a plusieurs solutions au problème mathématique avec des valeurs de t_exp négatives

Il faut :
- Enlever les outliers qui sont la résolution du problème mathémique mais pas au problème réel


"""
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, least_squares
from numpy.random import default_rng


def localisation_3c(c1, c2, c3, t1, t2, t3):
    """
    Localisation of the source with the first spike on each one of the three captors

    Parameters:
    ---------- 
    t1, t2, t3 : float
        Spikes on sensors 1, 2 and 3
    c1, c2, c3 : tuple (x : float, y: float)
        Localizations (xi, yi) of sensors
    
    Outputs
    -------
    x, y : int
        Localisation of the source
    """

    x1, y1 = c1
    x2, y2 = c2
    x3, y3 = c3
    v = 340 # To edit, for the moment speed of the sound

    def func(p):
        x, y, t_exp = p
        return [(x - x1)**2 + (y - y1)**2 - v**2 * (t1 - t_exp)**2, (x - x2)**2 + (y - y2)**2 - v**2 * (t2 - t_exp)**2, (x - x3)**2 + (y - y3)**2 - v**2 * (t3 - t_exp)**2]


    score = np.inf

    for x0 in range(20, 80, 5):
        for y0 in range(20, 80, 5):
            root = least_squares(func, [x0, y0, 1], bounds = ((0, 0, -1),(110, 110, 1))) #Solve the equation

            if root.cost < score :
                x, y, t_exp = root.x

    return x, y


def affichage_3c(sensors, explosion):
    """
    Plot the sensors and the explosion localisations

    Parameters
    ----------
    sensors : list fo tuple of coordinates (x, y)
        List of coordinates of the sensors
    explosion : tuple (x, y)
        Localization of the explosion
    """

    x_sensors, y_sensors = list(zip(*sensors))
    
    plt.scatter(x_sensors, y_sensors, color = "black")
    plt.scatter(explosion[0], explosion[1], color = "red")
    plt.show()


def localisations(sensors, t, n = 10, show = True, additional_point = None, save_file = None):
    """
    Returns the localisation of the sources without any building.
    Eventually, plot the result.

    Parameters
    ----------
    sensors : list fo tuple of coordinates (x, y)
        List of coordinates of the sensors
    t : list of floats
        Time of first spike for each sensor
    n : int, default = 10
        Number of trio to take to have different possible source localisations
    show : bool, default = True
        If True, then plot the localisations of the possible explosions
    additional_point : None or tuple of coordinates (x, y)
        Additional plot that we want to plot

    Output
    ------
    local : list of tuple (x, y)
        Possible explosion sources
    """

    local = []

    x_sensors, y_sensors = list(zip(*sensors))
    plt.scatter(x_sensors, y_sensors, color = "black", marker = "P", s=40, label="Sensors")
    

    for _ in range(n):
        rng = default_rng()
        i, j, k = rng.choice(len(sensors), size=3, replace=False)
        c1, c2, c3 = sensors[i], sensors[j], sensors[k]
        t1, t2, t3 = t[i], t[j], t[k]
        localisation = localisation_3c(c1, c2, c3, t1, t2, t3)
        local.append(localisation)

    if show :
        plt.scatter(list(zip(*local))[0], list(zip(*local))[1], color = "red", s= 40, label="Possible sources")

    if additional_point :
        plt.scatter(additional_point[0], additional_point[1], marker='^', s = 80, label="Real source")

    if show :
        plt.legend()
        if save_file :
            plt.savefig(save_file)

        plt.show()


    return local


def first_spike(sensor_data, tresh=0.1,neighbourhood=3):
    """
    First spike detected by a sensor

    Parameters
    ----------
    sensor_data : Numpy array
        Time and corresponding signal measured by the sensor

    Output
    ------
    t : float
        Time of the first spike
    """

    signal_t = sensor_data['temps']
    signal_data = sensor_data['pression']
    local_max = []
    global_max = 0
    n = len(signal_data)
    for i in range(n):
        is_local_max = True
        for x in range(1,1+neighbourhood):
            if i-x >= 0:
                if signal_data[i-x] >= signal_data[i]:
                    is_local_max = False
                    #print(i,"not local because (x=" + str(x) + ") -- i is",signal_data[i],"i-x is",signal_data[i-x])
            if i+x < n:
                if signal_data[i+x] >= signal_data[i]:
                    is_local_max = False
                    
                    #print(i,"not local because (x=" + str(x) + ") -- i is",signal_data[i],"i+x is",signal_data[i+x])

        if is_local_max:
            local_max.append(i)
        if signal_data[i] > signal_data[global_max]:
            global_max = i


    for x in local_max:
        if signal_data[x] >= tresh*signal_data[global_max]:
            return signal_t[x]

def main(folder_stations, n = 10, explosion_source = None, save_file = None) :
    """
    Plot the real explosion source and n theoretical sources

    Parameters
    ----------
    folder_stations : string
        Folder where there are the sensors  
    sensors : list fo tuple of coordinates (x, y)
        List of coordinates of the sensors
    n : int, default = 10
        Number of trio to take to have different possible source localisations
    explosion_source : None or Tuple (x, y)
        If not None, plot the real source of the explosion
    """
    sensors_data = []
    sensors = []

    for filename in os.listdir(folder_stations):
        if filename != "STATION_NOM" :
            station_df = pd.read_csv(os.path.join(folder_stations, filename), sep = " ", index_col=False)
            sensors_data.append(station_df)
        
        else :
            with open(os.path.join(folder_stations, filename)) as f:
                lines = f.readlines()
            for line in lines:
                find = re.findall("ST[0-9]*", line)
                if find != []:
                    line = line.strip().split(" ")
                    x, y = float(line[1]), float(line[2])
                    sensors.append((x, y))
                    

    # First splike of each sensor
    t = []
    for sensor_data in sensors_data:
        t_detect = first_spike(sensor_data)
        t.append(t_detect)

    # Localisations
    _ = localisations(sensors= sensors, t = t, n = n, show = True, additional_point= explosion_source, save_file = save_file)

    

# Test
if __name__ == "__main__":
    
    # ------ Test 1 -----
    # c1, c2, c3, c4, c5 = (0,0), (2,0), (1,2), (3,4), (8,1)
    # t1, t2, t3, t4, t5 = 0, 0, 0, 0, 0
    # print(localisation_3c(c1, c2, c3, t1, t2, t3))

    # affichage_3c([c1, c2, c3], localisation_3c(c1, c2, c3, t1, t2, t3))
    # res = localisations([c1, c2, c3, c4, c5], [t1, t2, t3, t4, t5], n=5)

    # ----- Test 2 -----

    # data_sensor_0 = pd.read_csv("Simulation_1/STATION_ST0", sep = " ", index_col=False)
    # f_spike = first_spike(data_sensor_0)
    # print(f_spike)

    # ----- Test 3 : Without building-----
    explosion = (40, 65)
    main(folder_stations="Simulations/Simu_without_building_40_65", n = 20, explosion_source=explosion)

    # ----- Test 4 : With building-----
    explosion = (40, 65)
    main(folder_stations="Simulation_2", n = 20, explosion_source=explosion)


    print("Everything seems to work ...")




