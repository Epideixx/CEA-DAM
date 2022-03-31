from asyncore import write
import numpy as np
import csv
import argparse
from fonctionCoutPierre.differentialTimeWarping import displayMatch, costDDTW

#from differentialTimeWarping import displayMatch

deltaT = 9.03696114115064 * 1e-5
nb_stations = 5


def getData(folder):

    # fig, axs = plt.subplots(nb_stations, sharey=True)
    data_t = [[] for i in range(nb_stations)]
    data_sign = [[] for i in range(nb_stations)]

    for index in range(nb_stations):
        file_name = "../" + folder + "/POST1D/TE/STATION_ST" + str(index)

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

    #for i in range(nb_stations):
    #    # axs[i].plot(data_t[i], data_sign[i])
    #    plt.plot(data_t[i], data_sign[i], label='station' + str(i))
    #    #plt.legend()
    # #plt.show()

    signalArray = np.empty((nb_stations, len(data_t[0]), 2))

    for k in range(nb_stations):
        signalArray[k, :, 0] = data_t[k]
        signalArray[k, :, 1] = data_sign[k]

    signalArray[:, :, 1] -= 1e5
    return signalArray

def correlation_lags(N):
    l = [i for i in range(N)]
    lm = list(reversed([i+1 for i in range(N-1)]))
    return np.array(lm + l)

def correlationSignals(signalArray1, signalArray2):
    # Calcul les intercorrelations des signaux reçus stations par stations
    # Renvoi aussi les autocorrelation du signal1 considéré comme le signal d'origine

    #fig, axs = plt.subplots(nrows=nb_stations, ncols=2)

    sig1k = signalArray1[0]
    sig2k = signalArray2[0]

    N = np.min((len(sig1k), len(sig2k)))
    # On prend la taille minimale des signaux pour tronquer les autres signaux
    # Car pour deux explosions différentes, les signaux ne seront pas de la même longueur.

    lags = deltaT * \
        correlation_lags(N, N)
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

        #axs[k, 0].plot(
        #    lags, interCorr, label='correlation entre les signaux de la station' + str(k))
        #axs[k, 0].legend()

        #axs[k, 1].plot(sig1k[:, 0], sig1k[:, 1],
        #               label='signal 60 station' + str(k))
        #axs[k, 1].plot(sig2k[:, 0], sig2k[:, 1],
        #               label='signal 50 station' + str(k))
        #axs[k, 1].legend()

        interCorrArray[k, :] = interCorr
        autoCorrArray[k, :] = autoCorr

    #plt.show()
    return interCorrArray, autoCorrArray, lags


def compareCorr(interCorrArray, autoCorrArray, lags):
    # Cette fonction compare les intercorrélations aux autocorrélations stations par stations
    # On fait la distance en norme 2 des inter/auto corrélations.
    # On somme toutes ces distances pour obtenir une distance totale

    corrDifference = interCorrArray - autoCorrArray
    distanceCorr = np.sum(np.linalg.norm(corrDifference, axis=1))

    #fig, axs = ##plt.subplots(nrows=nb_stations, sharex=True)

    #for k in range(nb_stations):

    #    axs[k].plot(lags, corrDifference[k],
    #                label='différence de corrélations pour la station' + str(k))
    #    axs[k].legend()

    #plt.show()
    print(np.sum(distanceCorr))
    return np.sum(distanceCorr)


# interArray, autoArray, lags = correlationSignals(signalArray40, signalArray60)
# compareCorr(interArray, autoArray, lags)

def are_close(x,y,precision = 0.000000001):
    return abs(x-y) < precision


def refine_time(sig1,sig2):
    time_list_1 = sig1[0,:,0]
    time_list_2 = sig2[0,:,0]

    nb_stations = 1

    full_time_list = np.union1d(time_list_1,time_list_2)


    new_sig1 = np.zeros((nb_stations,np.shape(full_time_list)[0],2))
    new_sig2 = np.zeros((nb_stations,np.shape(full_time_list)[0],2))


    ind1 = 0
    ind2 = 0
    ind_t = 0
    block_1 = False
    block_2 = False


    t1 = time_list_1[0]
    t2 = time_list_2[0]



    for ind_t in range(len(full_time_list)):
        t = full_time_list[ind_t]
        print(t,t1,t2,ind1,ind2,are_close(t1,t),are_close(t2,t))

        if are_close(t1,t):
            for i in range(nb_stations):
                new_sig1[i,ind_t] = np.array((t,sig1[i,ind1,1]))
        else:
            for i in range(nb_stations):
                if ind1 < np.shape(time_list_1)[0] - 1:
                    nv =  np.array((t,(sig1[i,ind1-1,1] + (sig1[i,ind1,1] - sig1[i,ind1-1,1])*(t-t1)/(time_list_1[ind1+1]-t1))))
                    #print("nv1", nv)
                    new_sig1[i,ind_t] = nv
                else:
                    new_sig1[i,ind_t] = np.array((t,sig1[i,ind1,1]))
             

        if are_close(t2,t):
            for i in range(nb_stations):
                new_sig2[i,ind_t] = np.array((t,sig2[i,ind2,1]))
        else:
            for i in range(nb_stations):
                if ind2 < np.shape(time_list_2)[0] - 1:
                    nv = np.array((t,(sig2[i,ind2-1,1] + (sig2[i,ind2,1] - sig2[i,ind2-1,1])*(t-t2)/(time_list_2[ind2+1]-t2))))
                    #print("nv2",nv)
                    new_sig2[i,ind_t] = nv
                else:
                    new_sig2[i,ind_t] = np.array((t,sig2[i,ind2,1]))
        if ind_t < len(full_time_list)-1:            
            if ind1 < len(time_list_1):
                if are_close(time_list_1[ind1+1],full_time_list[ind_t+1]):
                    ind1 += 1
                    t1 = time_list_1[ind1]
            if ind2 < len(time_list_2):
                if are_close(time_list_2[ind2+1],full_time_list[ind_t+1]):
                    ind2 += 1
                    t2 = time_list_2[ind2]

    return (new_sig1,new_sig2)


#l1 = np.zeros((1,5,2))
#l1[0,:,:] = np.array([(1,10),(1.5,12),(2,10),(3,12),(3.5,10)])
#l2 = np.zeros((1,5,2))
#l2[0,:,:] = np.array([(1,12),(1.5,10),(2.5,12),(2.75,10),(3.5,12)])

#time_list_1 = l1[0,:,0]

#time_list_2 = l2[0,:,0]
#full_time_list = np.union1d(time_list_1,time_list_2)
#print(full_time_list)

#print(refine_time(l1,l2))


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

    #fig, axs = #plt.subplots(nrows=nb_stations, ncols=2)
    # On affiche les signaux récalés par rapport aux signaux décalés
    for k in range(nb_stations):
        sig1NoDelayk = sigArrayNoDelay1[k, :, 1]
        sig2NoDelayk = sigArrayNoDelay2[k, :, 1]

        sig1k = sigArray1[k, :N, 1]
        sig2k = sigArray2[k, :N, 1]

    #    axs[k, 0].plot(time, sig1NoDelayk, label='No Delay')
    #    axs[k, 0].plot(time, sig2NoDelayk)
    #    axs[k, 0].legend()

    #    axs[k, 1].plot(time, sig1k, label='delay')
    #    axs[k, 1].plot(time, sig2k)
    #    axs[k, 1].legend()

    #plt.show()
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

    #fig, axs = plt.subplots(nrows=nb_stations)
    # On affiche les signaux originaux et retardés/Avancés

    #for k in range(nb_stations):
    #    axs[k].plot(sigArrayDelayed[k, :, 0],
    #                sigArrayDelayed[k, :, 1], label='delayed')
    #    axs[k].plot(sigArray[k, :, 0], sigArray[k, :, 1], label='original')
    #    axs[k].legend()

    #plt.show()
    return sigArrayDelayed


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

    # interArray, autoArray, lags = correlationSignals(
    #    sigArray1NoDelay, sigArray2NoDelay)

    #dist = compareCorr(interArray, autoArray, lags)

    path = []

    for i in range(nb_stations):
        path.append(displayMatch(sigArray1[i],sigArray2[i]))


    dist = 0

    for s in range(nb_stations):

        for couple in path[s]:
            (i,j) = couple

            dist += (sigArray1[s,i,0] - sigArray2[s,j,0])**2 + (sigArray1[s,i,1] - sigArray2[s,j,1])**2 

    dist /= (s*len(path[0]))

    return dist


#dist = cout(signalArray60, signalArray40)
#dist = cout(signalArray60, signalArray55)
#print(dist)
#dist = cout(signalArray60, sigArray60Delayed)

#signalArray60 = getData("TE6StatLoS")
#signalArray40 = getData('TEexplo40')
#signalArray55 = getData('TEexplo55')
#sigArray60Delayed = createDelay(signalArray60, 30)
#displayMatch(signalArray60[0], signalArray55[0])
#displayMatch(signalArray60[0], signalArray55[0], derivative=True)
