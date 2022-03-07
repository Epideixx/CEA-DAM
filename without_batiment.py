# ----------------------------------------------
#          Detect sources without buildings
# ----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.random import default_rng


# Y a une erreur dans les calculs ...
def localisation_3c(c1, c2, c3, t1, t2, t3):
    """
    Localisation of the source with the first spike on each one of the three captors
    Inputs : 
    - (t1, t2, t3) -> Spikes on sensors 1, 2 and 3
    - (c1, c2, c3) -> Localizations (xi, yi) of sensors
    Output : (x, y) = Localisation of the source
    """

    # With equation resolution
    x1, y1 = c1
    x2, y2 = c2
    x3, y3 = c3
    t_exp = 0
    v = 1

    def func(p):
        x, y, t_exp = p
        return [2*v**2*t_exp + 2*(x2 - x1)*x + 2*(y2 - y1)*y - (x2**2 + y2**2 - x1**2 - y1**2)+ v**2*(t2**2 - t1**2), 2*v**2*t_exp + 2*(x3 - x1)*x + 2*(y3 - y1)*y - (x3**2 + y3**2 - x1**2 - y1**2)+ v**2*(t3**2 - t1**2), 2*v**2*t_exp + 2*(x3 - x2)*x + 2*(y3 - y2)*y - (x3**2 + y3**2 - x2**2 - y2**2)+ v**2*(t3**2 - t2**2)]

    root = fsolve(func, [1, 1, 1])

    x, y, t_exp = root
    
    return x, y


def affichage_3c(sensors, explosion):
    """
    sensors -> list fo tuple of coordinates (x, y)
    explosion -> tuple (x, y) localization explosion
    """

    x_sensors, y_sensors = list(zip(*sensors))
    
    plt.scatter(x_sensors, y_sensors, color = "black")
    plt.scatter(explosion[0], explosion[1], color = "red")
    plt.show()


def localisation(sensors, t, n = 10):
    """
    n -> number of trio to take to have different possible source localisations
    """

    # A REFAIRE
    x_sensors, y_sensors = list(zip(*sensors))
    plt.scatter(x_sensors, y_sensors, color = "black")

    for _ in range(n):
        rng = default_rng()
        i, j, k = rng.choice(len(sensors), size=3, replace=False)
        c1, c2, c3 = sensors[i], sensors[j], sensors[k]
        t1, t2, t3 = t[i], t[j], t[k]
        localisation = localisation_3c(c1, c2, c3, t1, t2, t3)
        plt.scatter(localisation[0], localisation[1])

    plt.show()

if __name__ == "__main__":
    c1, c2, c3, c4, c5 = (0,0), (2,0), (1,2), (3,4), (8,1)
    t1, t2, t3, t4, t5 = 0, 0, 0, 0, 0
    print(localisation_3c(c1, c2, c3, t1, t2, t3))

    affichage_3c([c1, c2, c3], localisation_3c(c1, c2, c3, t1, t2, t3))
    localisation([c1, c2, c3, c4, c5], [t1, t2, t3, t4, t5], n=5)




# A CHECK


