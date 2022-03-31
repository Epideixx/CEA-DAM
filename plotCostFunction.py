import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from costDTW import costDDTW
from costMaxCorrelation import costCorrelation
from costSVD import interpSignals, costSVD
from buildings_data import BAT_list, SENSORS_list
from is_inside import is_inside_bat
from mpl_toolkits import mplot3d
from scipy import interpolate


nb_stations = 5
lbound = 20
ubound = 90
step = 10
nPoint = (ubound - lbound)//step


def getData(folder):

    # fig, axs = plt.subplots(nb_stations, sharey=True)
    data_t = [[] for i in range(nb_stations)]
    data_sign = [[] for i in range(nb_stations)]

    for index in range(nb_stations):
        file_name = "./" + folder + "/STATION_ST" + str(index)

        data = open(file_name, 'r')

        data_reader = csv.reader(data, delimiter=' ')

        first = True
        for row in data_reader:
            if not first:
                data_t[index].append(float(row[0]))
                data_sign[index].append(float(row[1]))
            else:
                first = False
        data.close()

    signalArray = np.empty((nb_stations, len(data_t[0]), 2))

    for k in range(nb_stations):
        signalArray[k, :, 0] = data_t[k]
        signalArray[k, :, 1] = data_sign[k]
        #plt.plot(data_t[k], data_sign[k])
    # plt.show()

    signalArray[:, :, 1] -= 1e5
    return signalArray


sigExplosion = getData("Cost_function_simulations/CAS_Simulation_sensor_50_60")
# sigExplosion2 = getData(
# "Cost_function_simulations/CAS_Simulation_sensor_0_0")


def plotCostFunction(sigExplosion, nb_stations, function='DDTW', NSVD=3000):

    x = np.arange(lbound, ubound, 10)
    X, Y = np.meshgrid(x, x)
    Z = np.zeros((nPoint, nPoint))

    if function == 'SVD':
        T = np.linspace(0, 1, NSVD)
        sigExplosionInterp = interpSignals(sigExplosion, NSVD)
        sigSimu = np.full((nb_stations, nPoint**2, NSVD), fill_value=0)

    for i in range(lbound, ubound, step):
        for j in range(lbound, ubound, step):
            if is_inside_bat((i, j), BAT_list):
                Z[(i-lbound)//step, (j-lbound)//step] = - np.inf
            else:
                sigLocal = getData(
                    "Cost_function_simulations/CAS_Simulation_sensor_" + str(i) + "_" + str(j))
                if function == 'DDTW':
                    cost = costDDTW(sigExplosion,
                                    sigLocal, nb_stations)
                    Z[(i-lbound)//step, (j-lbound)//step] = cost
                    print(i, j, cost)
                elif function == 'correlation':
                    cost = costCorrelation(sigExplosion,
                                           sigLocal, nb_stations)
                    Z[(i-lbound)//step, (j-lbound)//step] = cost
                    print(i, j, cost)
                elif function == 'SVD':
                    sigSimu[:, nPoint*(i-lbound)//step+(j-lbound)//step,
                            :] = interpSignals(sigLocal, NSVD)

    if function == 'SVD':
        cost = costSVD(nb_stations, sigExplosionInterp, sigSimu, nSingular=5)
        cost = np.reshape(cost, (nPoint, nPoint))
        Z = cost
    M = np.max(Z)
    Z[Z == -np.inf] = M

    with open('Z.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(nPoint):
            row = Z[i, :]
            writer.writerow(row)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    print(Z)
    plt.show()
    return Z


#Z = plotCostFunction(sigExplosion, nb_stations)

#print(costDDTW(sigExplosion, sigExplosion2, nb_stations))


def plotPoly(poly=True, polyStep=step, log=True):

    with open('Z.csv', 'r') as f:
        Z = np.zeros((nPoint, nPoint))
        x = np.arange(lbound, ubound, step)
        X, Y = np.meshgrid(x, x)

        reader = csv.reader(f, delimiter=',')
        i = 0
        for line in reader:
            if i % 2 == 0:
                for j in range(nPoint):
                    Z[i//2, j] = float(line[j])
            i += 1

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        if poly:
            if log:
                f = interpolate.interp2d(X, Y, np.log(Z+1), kind='cubic')
            else:
                f = interpolate.interp2d(X, Y, Z, kind='cubic')

            xnew = np.arange(lbound+10, ubound, polyStep)
            znew = f(xnew, xnew)
            Xnew, Ynew = np.meshgrid(xnew, xnew)
            ax.plot_surface(Xnew, Ynew, znew, cmap='jet')

        else:
            if log:
                ax.plot_surface(X, Y, np.log(Z+1), cmap='jet')
            else:
                ax.plot_surface(X, Y, Z, cmap='jet')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.zaxis.set_scale('log')
        # print(Z)
        plt.show()


Z = plotCostFunction(sigExplosion, nb_stations, function='correlation')

#print(costDDTW(sigExplosion, sigExplosion2, nb_stations))

plotPoly(polyStep=1, log=False)
