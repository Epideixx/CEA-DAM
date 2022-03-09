#!"C:\Python33\python.exe"
import matplotlib.pyplot as plt
import csv
import argparse
COLORS = ['blue','red','green','orange','purple','grey']

parser = argparse.ArgumentParser(description="retrieve file name")

parser.add_argument("--folder", help="Folder where station data is to be read", required=True)

args = parser.parse_args()

nb_stations = 5
# fig,axs = plt.subplots(nb_stations, sharey = True)
data_t = [[] for i in range(nb_stations)]
data_sign = [[] for i in range(nb_stations)]

for index in range(nb_stations):
    file_name = args.folder + "\STATION_ST" + str(index)

    

    data = open(file_name,'r')
    
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
    # axs[i].plot(data_t[i],data_sign[i])
    plt.plot(data_t[i], data_sign[i], color=COLORS[i])
plt.show()


# python plot_multi_data.py --folder folder_name