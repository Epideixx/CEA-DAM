from re import A
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

nb_stations = 5

'''S = np.empty((25, 1000))
for i in range(25):
    s = np.sin(np.linspace(0, np.pi, 1000) + i*np.pi/12)
    S[i] = s


svd = TruncatedSVD(n_components=2)

svd.fit(S)
print(svd.explained_variance_ratio_, 'ratio')
print(svd.singular_values_, 'singular values')
print(svd.components_, np.shape(svd.components_.transpose()), 'components')
print(np.matmul(S, svd.components_.transpose()), 'vectors')

Sr = np.matmul(S, svd.components_.transpose())

for k in range(25):
    plt.scatter(Sr[k, 0], Sr[k, 1], label=str(k))
plt.legend()
plt.show()'''


def interpSignals(sigArray, NSVD=3000):
    """
    piecewise linear interp of nb_stations time series

    Parameters:
    ----------
    nb_stations : int
        number of stations
    NSVD : int
        size of interpolation


    Outputs
    -------
    sigInterpolated : numpy ndarray
        shape(nb_stations, NSVD) signals interpolated.
    """
    nb_stations = np.shape(sigArray)[0]
    N = np.shape(sigArray)[1]

    sigInterpolated = np.empty((nb_stations, NSVD))

    for k in range(nb_stations):
        sigInterpolated[k] = np.interp(np.linspace(
            0, 1, NSVD), sigArray[k, :, 0], sigArray[k, :, 1])

    return sigInterpolated


def costSVD(nb_stations, sigExplosion, sigSimu, nSingular):
    """
    Calculates distance between all groups of signals (all stations for a simulation)

    Parameters:
    ----------
    nb_stations : int
        number of stations
    sigExplosion : numpy ndarray
        shape(nb_stations, N) signals for the original explosion. N is the max size of a signal.
    sigSimu : numpy ndarray
        shape(nb_stations, nb_simus, N) array of all other signals.
    nSingular : int
        number of singular values computed. Cannot exceed the number of signals


    Outputs
    -------
    costSVD : numpy ndarray
        shape(nb_simus)
    """

    nb_simus = np.shape(sigSimu)[1]
    N = np.shape(sigSimu)[2]
    sigExploLocal = np.empty((1, N))

    distances = np.empty((nb_stations, nb_simus))

    for n in range(nb_stations):
        S = sigSimu[n]
        sigExploLocal[0] = sigExplosion[n]
        S = np.concatenate((S, sigExploLocal), axis=0)

        svd = TruncatedSVD(n_components=nSingular)
        svd.fit(S)
        Sr = np.matmul(S, svd.components_.transpose())

        for k in range(nb_simus):
            vector = Sr[-1] - Sr[k]
            distanceLocal = np.linalg.norm(vector)
            distances[n, k] = distanceLocal

    costSVD = np.mean(distances, axis=0)
    return costSVD
