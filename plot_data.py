import matplotlib.pyplot as plt
import csv
import argparse

parser = argparse.ArgumentParser(description="retrieve file name")

parser.add_argument("--file", help="File to be read", required=True)

args = parser.parse_args()

with open(args.file) as data:
    data_reader = csv.reader(data, delimiter=' ')
    data_t = []
    data_sign = []
    first = True
    for row in data_reader:
        if not first:
            data_t.append(float(row[0]))
            data_sign.append(float(row[1]))
        else:
            first = False
    
    plt.scatter(data_t,data_sign)
    plt.show()

### python plot_data --file file_name