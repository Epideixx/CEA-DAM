# -------------------------------------
#             Speed estimation
# -------------------------------------

from without_building import first_spike, localisations, main
import os
import numpy as np
from scipy.optimize import curve_fit
import csv
import pandas as pd
import re
import math

def speed(folder_name : str):
    """
    Parameters
    ----------
    folder_name : str
        Folder wontaining every simulation

    Returns
    -------
    Speed : dict = {explosion_energy : {Vs : Parameter Vs, a : Parameter a, alpha : Parameter alpha, beta : Parameter beta}}
        Speed parameters according to the paper found, obtained with minimisation, and depending of the explosion energy
    """
    simu_energies = list(os.listdir(folder_name))
    speed = {}

    for folder_name in simu_energies:
        temp = re.findall(r'\d+', folder_name)
        energy = list(map(int, temp))

        speed[energy : speed_for_energy(folder_name)]

    return speed

def speed_for_energy(folder):
    """
    Parameters
    ----------
    folder : str 
        Name of the folder with the sensors

    Returns
    -------
    Speed : {Vs : Parameter Vs, a : Parameter a, alpha : Parameter alpha, beta : Parameter beta}
        Speed parameters according to the paper found, obtained with minimisation, and depending of the explosion energy
    """

    # Remind : There are a simulation for each energy with a lot of sensors
    
    nb_stations = len(os.listdir(folder)) - 1

    localisations = []
    file_name = os.path.join(folder,"STATION_NOM")
    
    data = open(file_name,'r')
    data_reader = csv.reader(data, delimiter=' ')
        
    for i, row in enumerate(data_reader):
        if i == 1:
            nb_stations = int(row[0])
        elif i > 2 and i<= 2 + nb_stations:
            localisations.append((float(row[1]), float(row[2])))
            
    measures = []

    for index in range(nb_stations):
        file_name = os.path.join(folder,"STATION_ST" + str(index))

        with open(file_name) as data:
            data_reader = csv.reader(data, delimiter=' ')
            data_t = []
            data_sign = []
            first = True
            for row in data_reader:
                if not first:
                    data_t.append(float(row[0]))
                    data_sign.append(float(row[1]))
                else:
                    first = False
            
            d = {"temps" : data_t, "pression" : data_sign}
            measures.append(pd.DataFrame(d))

    speed = speed_sensors(measures=measures, localisations=localisations)

    return speed


def speed_sensors(measures, localisations):
    """
    Parameters
    ----------
    measures : list of Pandas
        Arrays representing (time, pressure)
    localisations : (tuple) 
        localisations of the sensors

    Returns
    -------
    Speed : {Vs : Parameter Vs, a : Parameter a, alpha : Parameter alpha, beta : Parameter beta}
        Speed parameters according to the paper found, obtained with minimisation, and depending of the explosion energy
    """

    t_explosion = np.inf
    t = []

    for i, (x, y) in enumerate(localisations):
        t_station = first_spike(measures[i])
        if t_station < t_explosion:
            t_explosion = t_station
            x_exp = x
            y_exp = y
        t.append(t_station)

    t = [t_station - t_explosion for t_station in t]

    dist = [np.sqrt((x - x_exp)**2 + (y - y_exp)**2) for (x, y) in localisations]

    v = [dist[i]/(t[i] + 10**(-10)) for i in range(len(t))]

    speed = fit(t = t, v_to_fit=v)

    return speed
    

def fit(t : list, v_to_fit : list):
    """
    Parameters
    ----------
    t : list of floats
        Time when signal is received by each sensor
    v_to_fit : list of floats 
        Speed to fit

    Returns
    -------
    Speed : {Vs : Parameter Vs, a : Parameter a, alpha : Parameter alpha, beta : Parameter beta}
        Speed parameters according to the paper found, obtained with minimisation, and depending of the explosion energy
    """

    def func(t, Vs, a, alpha, beta):
        return Vs*np.exp(-(alpha*np.array(t)))*(1 - beta*np.array(t)) + a*np.log(1 + beta * np.array(t))

    parameters, autre = curve_fit(func, t ,v_to_fit)

    print(parameters)
    print(autre)
    return {"Vs"  : parameters[0], "a" : parameters[1], "alpha" : parameters[2], "beta" : parameters[3]}


if __name__ == "__main__":
    folder = os.path.join("Simulations", "Simu_with_building_40_25")
    speed = speed_for_energy(folder)
    print(speed)