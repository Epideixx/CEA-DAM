import numpy as np


nb_stations = 5


def normalize(signal):
    norm = np.linalg.norm(signal[:, 1])
    signal[:, 1] /= norm


def costCorrelation(sigArray1, sigArray2, nb_stations=nb_stations):
    """
    calculates the cost between the signals of the explosion and the signals of the smulations
        by computing the maximum value of the correlations between the signals

    Parameters:
    ---------- 
    sigArray1 : numpy ndarray
        shape(nb_stations, signal_length, 2) : index 0 is time, index 1 are pressure values
        supposed to represent the real explosion
    sigArray2 : numpy ndarray
        shape(nb_stations, signal_length, 2) : index 0 is time, index 1 are pressure values
    nb_stations : int
        number of stations


    Outputs
    -------
    costSVD : float
        cost computed
    """
    for k in range(nb_stations):
        if (sigArray2[k, :, 1] == 0).all():
            sigArray2[k, :, 1] += 0.1
        normalize(sigArray1[k])
        normalize(sigArray2[k])
    maxCorrelations = np.empty(nb_stations)

    for k in range(nb_stations):
        sig1k = sigArray1[k]
        sig2k = sigArray2[k]

        interCorr = np.correlate(np.abs(sig1k[:, 1]), np.abs(sig2k[:, 1]))

        maxCorrelations[k] = np.max(interCorr)
    cost = 1 - np.mean(maxCorrelations)
    return cost
