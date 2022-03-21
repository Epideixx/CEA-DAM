import numpy as np
# import matplotlib.pyplot as plt
import csv
import argparse
from scipy import signal as sgn
from tester import displayMatch

deltaT = 9.03696114115064 * 1e-5
nb_stations = 6


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

    # for i in range(nb_stations):
        # axs[i].plot(data_t[i], data_sign[i])
        # plt.plot(data_t[i], data_sign[i], label='station' + str(i))
        # plt.legend()
    # plt.show()

    signalArray = np.empty((nb_stations, len(data_t[0]), 2))

    for k in range(nb_stations):
        signalArray[k, :, 0] = data_t[k]
        signalArray[k, :, 1] = data_sign[k]

    signalArray[:, :, 1] -= 1e5
    return signalArray


signalArray60 = getData("TE6StatLoS")
signalArray40 = getData('TEexplo40')
signalArray55 = getData('TEexplo55')


def correlationSignals(signalArray1, signalArray2):
    # Calcul les intercorrelations des signaux reçus stations par stations
    # Renvoi aussi les autocorrelation du signal1 considéré comme le signal d'origine

    # fig, axs = plt.subplots(nrows=nb_stations, ncols=2)

    sig1k = signalArray1[0]
    sig2k = signalArray2[0]

    N = np.min((len(sig1k), len(sig2k)))
    # On prend la taille minimale des signaux pour tronquer les autres signaux
    # Car pour deux explosions différentes, les signaux ne seront pas de la même longueur.

    lags = deltaT * \
        sgn.correlation_lags(N, N)
    # Calcul des retards dans les corrélations

    L = len(lags)

    interCorrArray = np.empty((nb_stations, L))
    autoCorrArray = np.empty((nb_stations, L))
    # On initialise les tableaux d'intercorrélations et d'autocorrélations

    for k in range(nb_stations):
        sig1k = signalArray1[k, :N]
        sig2k = signalArray2[k, :N]

        interCorr = sgn.correlate(np.abs(sig1k[:, 1]), np.abs(sig2k[:, 1]))
        autoCorr = sgn.correlate(np.abs(sig1k[:, 1]), np.abs(sig1k[:, 1]))

        # axs[k, 0].plot(
        #    lags, interCorr, label='correlation entre les signaux de la station' + str(k))
        # axs[k, 0].legend()

        # axs[k, 1].plot(sig1k[:, 0], sig1k[:, 1],
        # label = 'signal 60 station' + str(k))
        # axs[k, 1].plot(sig2k[:, 0], sig2k[:, 1],
        # label='signal 50 station' + str(k))
        # axs[k, 1].legend()

        interCorrArray[k, :] = interCorr
        autoCorrArray[k, :] = autoCorr
    #axs[0, 0].legend()
    #axs[0, 1].legend()

    # plt.show()
    return interCorrArray, autoCorrArray, lags


def compareCorr(interCorrArray, autoCorrArray, lags):
    # Cette fonction compare les intercorrélations aux autocorrélations stations par stations
    # On fait la distance en norme 2 des inter/auto corrélations.
    # On somme toutes ces distances pour obtenir une distance totale

    corrDifference = interCorrArray - autoCorrArray
    distanceCorr = np.sum(np.linalg.norm(corrDifference, axis=1))

    # fig, axs = plt.subplots(nrows=nb_stations, sharex=True)

    # for k in range(nb_stations):
    #
    #    axs[k].plot(lags, corrDifference[k],
    #                label='différence de corrélations pour la station' + str(k))
    #    axs[k].legend()

    # plt.show()
    print(np.sum(distanceCorr))
    return np.sum(distanceCorr)


# interArray, autoArray, lags = correlationSignals(signalArray40, signalArray60)
# compareCorr(interArray, autoArray, lags)


def getDelay(signal1Stat1, signal2Stat1):
    # le signal 1 est l'original
    # détermine le retard d'un signal sur l'autre
    # On calcul spécifiquement le retard et on donne quel signal est en retard sur l'autre
    sig1 = signal1Stat1[:, 1]
    sig2 = signal2Stat1[:, 1]
    firstMax1 = np.argmax(sig1 > 1000)
    firstMax2 = np.argmax(sig2 > 1000)

    delay = firstMax1 - firstMax2
    # le retard est calculé en terme d'indice de liste
    # On utilise un seuil d'une valeur de 1000 Pascal pour déterminer l'indice de début d'un signal

    if delay > 0:
        delayed = 1
    if delay <= 0:
        delayed = 2
    # On écrit quel signal est en retard sur l'autre

    return np.abs(delay), delayed


def removeDelay(sigArray1, sigArray2):
    # Recadre les signaux de toutes les stations par rapport au retard des signaux de la station 0

    sigToCompare1 = sigArray1[0]
    sigToCompare2 = sigArray2[0]
    # on isole les signaux de la station 0

    N = np.min((len(sigToCompare1), len(sigToCompare2)))
    delay, delayed = getDelay(sigToCompare1, sigToCompare2)
    time = sigArray1[0, :N, 0]
    # On récupere le retard et quel signal est retardé.
    # On calcul un indice max N qui correspond à une troncature

    sigArrayNoDelay1 = np.zeros((nb_stations, N, 2))
    sigArrayNoDelay2 = np.zeros((nb_stations, N, 2))
    # On initiatlise les tableaux de signaux dont on aura retiré le retard.

    if delayed == 1:
        for k in range(nb_stations):
            sigArrayNoDelay1[k, :N-delay, 1] = sigArray1[k, delay:N, 1]
            # Si le premier signal est retardé, on le réécrit à partir d'indices plus tardifs
            # le reste des valeurs à gauche du signal seront des 0
            sigArrayNoDelay2[k, :, 1] = sigArray2[k, :N, 1]
            sigArrayNoDelay1[k, :, 0] = time
            sigArrayNoDelay2[k, :, 0] = time
    else:
        for k in range(nb_stations):
            sigArrayNoDelay1[k, delay:, 1] = sigArray1[k, :N - delay, 1]
            # Si c'est le second signal qui est retardé, on réécrit le premier signal plus loin dans le tableau
            # le reste des valeurs à gauche du signal seront des 0
            sigArrayNoDelay2[k, :, 1] = sigArray2[k, :N, 1]
            sigArrayNoDelay1[k, :, 0] = time
            sigArrayNoDelay2[k, :, 0] = time

            '''sigArrayNoDelay2[k, :N-delay, 1] = sigArray2[k, delay:N, 1]
            sigArrayNoDelay1[k, :, 1] = sigArray1[k, :N, 1]
            sigArrayNoDelay1[k, :, 0] = time
            sigArrayNoDelay2[k, :, 0] = time'''

    # fig, axs = plt.subplots(nrows=nb_stations, ncols=2)
    # On affiche les signaux récalés par rapport aux signaux décalés
    #   for k in range(nb_stations):
    #       sig1NoDelayk = sigArrayNoDelay1[k, :, 1]
    #       sig2NoDelayk = sigArrayNoDelay2[k, :, 1]
    #
    #       sig1k = sigArray1[k, :N, 1]
    #       sig2k = sigArray2[k, :N, 1]
    #
    #       axs[k, 0].plot(time, sig1NoDelayk, label='No Delay')
    #       axs[k, 0].plot(time, sig2NoDelayk)
    #       axs[k, 0].legend()
    #
    #       axs[k, 1].plot(time, sig1k, label='delay')
    #       axs[k, 1].plot(time, sig2k)
    #       axs[k, 1].legend()
    #
    #   plt.show()
    return sigArrayNoDelay1, sigArrayNoDelay2


# removeDelay(signalArray60, signalArray40)


def createDelay(sigArray, delay):
    # delay est un entier indice de liste pour l'instant
    # Crée artificiellement un retard dans un signal pour voir si la fonction de cout
    # est robuste au retard.

    time = sigArray[0, :, 0]
    N = len(time)
    sigArrayDelayed = np.zeros((nb_stations, N, 2))
    # On initialise le tableau de signaux retardés

    if delay >= 0:
        for k in range(nb_stations):
            sigk = sigArray[k, :, 1]
            sigArrayDelayed[k, delay:, 1] = sigk[:N-delay]
            sigArrayDelayed[k, :, 0] = sigArray[k, :, 0]
            # si c'est un retard, on fait un certain décalage

    if delay < 0:
        for k in range(nb_stations):
            sigk = sigArray[k, :, 1]
            sigArrayDelayed[k, :N+delay, 1] = sigk[-delay:]
            sigArrayDelayed[k, :, 0] = sigArray[k, :, 0]
            # si c'est une avance on fait un autre décalage

    '''fig, axs = plt.subplots(nrows=nb_stations)
    # On affiche les signaux originaux et retardés/Avancés

    for k in range(nb_stations):
        axs[k].plot(sigArrayDelayed[k, :, 0],
                    sigArrayDelayed[k, :, 1], label='delayed')
        axs[k].plot(sigArray[k, :, 0], sigArray[k, :, 1], label='original')
        axs[k].legend()

    plt.show()'''
    return sigArrayDelayed


sigArray60Delayed = createDelay(signalArray60, -30)

# a, b = removeDelay(signalArray60, sigArray60Delayed)


# sigArray60NoDelay, sigArray55NoDelay = removeDelay(
# signalArray60, signalArray55)
# interArray, autoArray, lags = correlationSignals(
# sigArray55NoDelay, sigArray60NoDelay)
# dist = compareCorr(interArray, autoArray, lags)


def cout(sigArray1, sigArray2):
    # calcul la distance entre les signaux issus de deux explosions
    sigArray1NoDelay, sigArray2NoDelay = removeDelay(
        sigArray1, sigArray2)

    interArray, autoArray, lags = correlationSignals(
        sigArray1NoDelay, sigArray2NoDelay)

    dist = compareCorr(interArray, autoArray, lags)

    return dist


# dist = cout(signalArray60, signalArray40)
# dist = cout(signalArray60, signalArray55)
# dist = cout(signalArray60, sigArray60Delayed)


def normalize(signal):
    norm = np.linalg.norm(signal[:, 1])
    signal[:, 1] /= norm


def coutMaxCorrelation(sigArray1, sigArray2):
    for k in range(nb_stations):
        normalize(sigArray1[k])
        normalize(sigArray2[k])
    #fig, axs = plt.subplots(nrows=nb_stations, ncols=2)

    sig1k = sigArray1[0]
    sig2k = sigArray2[0]

    N = np.min((len(sig1k), len(sig2k)))
    # On prend la taille minimale des signaux pour tronquer les autres signaux
    # Car pour deux explosions différentes, les signaux ne seront pas de la même longueur.

    lags = deltaT * \
        sgn.correlation_lags(N, N)
    # Calcul des retards dans les corrélations

    L = len(lags)

    # On initialise les tableaux d'intercorrélations et d'autocorrélations

    maxCorrelations = np.empty(nb_stations)

    for k in range(nb_stations):
        sig1k = sigArray1[k, :N]
        sig2k = sigArray2[k, :N]

        interCorr = sgn.correlate(np.abs(sig1k[:, 1]), np.abs(sig2k[:, 1]))

        '''axs[k, 0].plot(
            lags, interCorr, label='correlation entre les signaux de la station ' + str(k))
        axs[k, 0].legend()

        axs[k, 1].plot(sig1k[:, 0], sig1k[:, 1],
                       label='signal original station ' + str(k))
        axs[k, 1].plot(sig2k[:, 0], sig2k[:, 1],
                       label='signal autre explosion station ' + str(k))
        axs[k, 1].legend()'''

        maxCorrelations[k] = np.max(interCorr)
    cost = 1 - np.mean(maxCorrelations)
    # plt.show()
    print(cost)


# coutMaxCorrelation(signalArray60, signalArray40)
# coutMaxCorrelation(signalArray60, signalArray55)
# coutMaxCorrelation(signalArray60, sigArray60Delayed)


displayMatch(sigArray60Delayed[0], signalArray60[0])
displayMatch(sigArray60Delayed[0], signalArray60[0], derivative=True)
