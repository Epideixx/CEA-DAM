

def getData(folder, nb_stations):

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

    for i in range(nb_stations):
        # axs[i].plot(data_t[i], data_sign[i])
        plt.plot(data_t[i], data_sign[i], label='station' + str(i))
        plt.legend()
    plt.show()

    signalArray = np.empty((nb_stations, len(data_t[0]), 2))

    for k in range(nb_stations):
        signalArray[k, :, 0] = data_t[k]
        signalArray[k, :, 1] = data_sign[k]

    signalArray[:, :, 1] -= 1e5
    return signalArray
