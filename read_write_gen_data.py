import numpy as np

def write_in_txt(dhist,fname):
    f = open("gen_data/" + fname, "w")
    for g in range(0,len(dhist)):
        gen_list = dhist[g]
        f.write("> GENERATION {0}\n".format(g+1))
        for (ax,ay,cost_a) in gen_list:
            f.write("{0};{1};{2}\n".format(ax,ay,cost_a))
        f.write("\n")
    f.close()

def read_in_txt(fname):
    dhist = [] ; gen_data = [[],[],[]]
    f = open("gen_data/" + fname, "r")
    txt_lines = f.read().split("\n")
    for line in txt_lines:
        if len(line)>0:
            if (line[0] == ">"): # new generation
                dhist.append(np.array(gen_data)) ; gen_data = [[],[],[]]
            else:
                elts = line.split(";")
                for i in [0,1,2]:
                    gen_data[i].append(float(elts[i]))
    f.close()
    return(dhist[1:])