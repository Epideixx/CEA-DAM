import numpy as np
from math import pi
import matplotlib.pyplot as plt


def derivativeEstimation(sig):
    derivative = np.empty(len(sig))
    derivative[1:-1] = (sig[1:-1] - sig[0: -2] + (sig[2:] - sig[0: -2])/2)/2
    derivative[0] = derivative[1]
    derivative[-1] = derivative[-2]
    return derivative


def DDTW(sig1, sig2):
    n = len(sig1)
    m = len(sig2)

    values1 = sig1[:, 1]
    values2 = sig2[:, 1]

    derivative1 = derivativeEstimation(values1)
    derivative2 = derivativeEstimation(values2)

    DDTW = np.full((n, m), np.inf)
    print(n, m)
    for i in range(n):
        for j in range(m):

            dist = (derivative1[i] - derivative2[j])**2
            if (i, j) == (0, 0):
                DDTW[i, j] = dist
            elif i == 0 and j != 0:
                DDTW[i, j] = dist + DDTW[i, j-1]
            elif j == 0 and i != 0:
                DDTW[i, j] = dist + DDTW[i-1, j]
            else:
                DDTW[i, j] = dist + \
                    min(DDTW[i-1, j-1], DDTW[i-1, j], DDTW[i, j-1])
    return DDTW, derivative1, derivative2


def getPath(DTW):
    n, m = np.shape(DTW)
    plotMatrix = np.full((n, m), False)

    i = n-1
    j = m-1

    path = [(n-1, m-1)]

    plotMatrix[n-1, m-1] = True

    while (i, j) != (0, 0):
        if i == 0:
            j -= 1
            plotMatrix[i, j] = True
        elif j == 0:
            i -= 1
            plotMatrix[i, j] = True
        else:
            sliceDTW = DTW[i-1:i+1, j-1:j+1]
            k, l = np.unravel_index(np.argmin(sliceDTW), (2, 2))
            i += k-1
            j += l-1
            plotMatrix[i, j] = True
        path.append(np.array([i, j]))
    return path[::-1]


def normalizeSignal(sig):

    interval = sig[-1, 0] - sig[0, 0]
    sig[:, 0] -= sig[0, 0]
    sig[:, 0] /= interval


def costDDTW(sigArray1, sigArray2, nb_stations):

    distances = np.empty(nb_stations)

    for k in range(nb_stations):

        sig1 = sigArray1[k]
        sig2 = sigArray2[k]
        warping, derivative1, derivative2 = DDTW(sig1, sig2)
        path = getPath(warping)

        distance = 0

        K = len(path)
        for l in range(K):
            index = path[l]
            i, j = index[0], index[1]
            # distance += np.sqrt((derivative1[i] - derivative2[j])
            # ** 2 + (sig1[i, 0] - sig2[j, 0])**2)
            distance += np.sqrt((derivative1[i] - derivative2[j])**2)
        distances[k] = distance/K
        #distances[k] = distance
    cost = np.mean(distances)

    return cost
