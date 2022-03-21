# -------------------------------------
#             Speed estimation
# -------------------------------------

from without_building import first_spike
import os
import numpy as np
from scipy.optimize import curve_fit

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
    pass


def speed_for_energy(energy, folder):
    """
    Parameters
    ----------
    energy : float
        Energy of the explosion
    folder : str 
        Name of the folder with the sensors

    Returns
    -------
    Speed : {Vs : Parameter Vs, a : Parameter a, alpha : Parameter alpha, beta : Parameter beta}
        Speed parameters according to the paper found, obtained with minimisation, and depending of the explosion energy
    """
    pass


def speed_sensors(measures, locations):
    """
    Parameters
    ----------
    measures : list of arrays/Pandas
        Arrays representing (time, pressure)
    locations : (tuple) 
        Locations of the sensors

    Returns
    -------
    Speed : {Vs : Parameter Vs, a : Parameter a, alpha : Parameter alpha, beta : Parameter beta}
        Speed parameters according to the paper found, obtained with minimisation, and depending of the explosion energy
    """

    t_explosion = np.inf
    t = []

    for i, (x, y) in enumerate(locations):
        t_station = first_spike(measures[i])
        if t_station < t_explosion:
            t_explosion = t_station
            x_exp = x
            y_exp = y
        t.append(t_station)

    t = [t_station - t_explosion for t_station in t]

    dist = [np.sqrt((x - x_exp)**2 + (y - y_exp)**2) for (x, y) in locations]

    v = [dist[i]/t[i] for i in range(len(t))]

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
        return Vs*np.exp(-(alpha*t))*(1 - beta*t) + a*np.log(1 + beta * t)

    parameters, _ = curve_fit(lambda Vs, a, alpha, beta : func(t, Vs, a, alpha, beta), v_to_fit) # Sans doute ne marchera pas parce que pb avec t à ne pas fit ...

    return {"Vs"  : parameters[0], "a" : parameters[1], "alpha" : parameters[2], "beta" : parameters[3]}