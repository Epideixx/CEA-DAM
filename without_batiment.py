# ----------------------------------------------
#          Detect sources without buildings
# ----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Y a une erreur dans les calculs ...
def localisation_3c(c1, c2, c3, t1, t2, t3):
    """
    Localisation of the source with the first spike on each one of the three captors
    Inputs : 
    - (t1, t2, t3) -> Spikes on sensors 1, 2 and 3
    - (c1, c2, c3) -> Localizations (xi, yi) of sensors
    Output : (x, y) = Localisation of the source
    """
    
    t = [t1, t2, t3]
    c = [c1, c2, c3]
    i_min = np.argmin(t)

    t_1 = t.pop(i_min)
    c_1 = c.pop(i_min)
    t_2, t_3 = t
    c_2, c_3 = c

    # With equation resolution
    B = np.array([[t_2 - t_1 - (c_2[0]**2 - c_1[0]**2) - (c_2[1]**2 - c_1[1]**2)],[t_3 - t_1 - (c_3[0]**2 - c_1[0]**2) - (c_3[1]**2 - c_1[1]**2)]])
    A_inv = 1/(2*((c_1[0] - c_2[0])*(c_1[1] - c_3[1]) - (c_1[0] - c_3[0])*(c_1[1] - c_2[1])))

    A_inv = A_inv * np.array([[(c_1[1] - c_3[1]), (c_2[1] - c_1[1])], [(c_3[0] - c_1[0]), (c_1[0] - c_2[0])]])

    x, y = np.dot(A_inv, B)[:, 0]
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
        i, j, k = np.random.randint(len(sensors), 3)
        c1, c2, c3 = sensors[i], sensors[j], sensors[k]
        t1, t2, t3 = t[i], t[j], t[k]
        localisation = localisation_3c(c1, c2, c3, t1, t2, t3)
        plt.scatter(localisation[0], localisation[1])

    plt.show

if __name__ == "__main__":
    c1, c2, c3 = (0,2), (0,0), (2,1)
    t1, t2, t3 = 0, 10, 00
    print(localisation_3c(c1, c2, c3, t1, t2, t3))

    affichage_3c([c1, c2, c3], localisation_3c(c1, c2, c3, t1, t2, t3))
    localisation([c1, c2, c3], [t1, t2, t3], n=1)







