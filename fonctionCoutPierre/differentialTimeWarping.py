import numpy as np
from math import pi
import matplotlib.pyplot as plt

n = 100
m = 100
lBound = 0
uBound = 4*pi
intervalLength = uBound - lBound


def carre(x):
    return x**2/100


vcarre = np.vectorize(carre)


#sig1 = np.sin(np.linspace(lBound, uBound, n))
time1 = np.linspace(lBound, uBound, n) + \
    np.random.normal(loc=0, scale=0.05, size=n)
time2 = np.linspace(lBound, uBound, m) + \
    np.random.normal(loc=0, scale=0.05, size=n)
values1 = np.sin(time1)
values2 = np.sin(time2)+1


sig1 = np.zeros((n, 2))
sig1[:, 0] = time1
sig1[:, 1] = values1

sig2 = np.zeros((m, 2))
sig2[:, 0] = time2
sig2[:, 1] = values2


def getStep(signal):
    return sig1[1, 0] - sig1[0, 0]


def derivativeEstimation(sig):
    derivative = np.empty(len(sig))
    derivative[1:-1] = (sig[1:-1] - sig[0: -2] + (sig[2:] - sig[0: -2])/2)/2
    derivative[0] = derivative[1]
    derivative[-1] = derivative[-2]
    return derivative


def DTW(sig1, sig2):
    n = len(sig1)
    m = len(sig2)

    values1 = sig1[:, 1]
    values2 = sig2[:, 1]

    DTW = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            dist = np.linalg.norm(
                [values1[i] - values2[j], sig1[i, 0]-sig2[j, 0]])
            if (i, j) == (0, 0):
                DTW[i, j] = dist
            elif i == 0 and j != 0:
                DTW[i, j] = dist + DTW[i, j-1]
            elif j == 0 and i != 0:
                DTW[i, j] = dist + DTW[i-1, j]
            else:
                DTW[i, j] = dist + min(DTW[i-1, j-1], DTW[i-1, j], DTW[i, j-1])
    return DTW


def DDTW(sig1, sig2):
    n = len(sig1)
    m = len(sig2)

    values1 = sig1[:, 1]
    values2 = sig2[:, 1]

    derivative1 = derivativeEstimation(values1)
    derivative2 = derivativeEstimation(values2)

    DDTW = np.full((n, m), np.inf)
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

    plt.imshow(plotMatrix)
    plt.show()
    path = np.array(path)

    return path[::-1, :]


def normalizeSignal(sig):
    '''mean = np.mean(sig[:, 1])
    sig[:, 1] -= mean
    std = np.linalg.norm(sig[:, 1])
    sig[:, 1] /= std'''

    interval = sig[-1, 0] - sig[0, 0]
    sig[:, 0] -= sig[0, 0]
    sig[:, 0] /= interval


def displayMatch(sig1, sig2, derivative=False):
    normalizeSignal(sig1)
    normalizeSignal(sig2)
    if derivative:
        a, d1, d2 = DDTW(sig1, sig2)
        print('derivative method', a[-1, -1])
    else:
        a = DTW(sig1, sig2)
        print('regular method', a[-1, -1])
    path = getPath(a)
    K = len(path)
    m = len(sig2)
    #print('warping', (K-m)/m)

    #for k in range(K):
    #    i = path[k, 0]
    #    j = path[k, 1]
        #plt.plot([sig1[i, 0], sig2[j, 0]], (sig1[i, 1], sig2[j, 1]),
    #             color='blue')
    
    #plt.plot(sig1[:, 0], sig1[:, 1], color='red')
    #plt.plot(sig2[:, 0], sig2[:, 1], color='red')
    #plt.show()

    return path


#displayMatch(sig1, sig2)
#displayMatch(sig1, sig2, derivative=True)

nb_stations = 6


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
            distance += (derivative1[i] - derivative2[j])**2

        distances[k] = distance

    cost = np.mean(distances)

    return cost
