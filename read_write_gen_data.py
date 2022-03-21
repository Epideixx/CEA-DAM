import numpy as np

def write_in_txt(dhist,fname):
    """ Write full genetic history in fname (list of (pts,cost) in each generations). """
    f = open("gen_data/" + fname, "w")
    for g in range(0,len(dhist)):
        gen_list = dhist[g]
        f.write("> GENERATION {0}\n".format(g+1))
        for (ax,ay,cost_a) in gen_list:
            f.write("{0};{1};{2}\n".format(ax,ay,cost_a))
        f.write("\n")
    f.close()

def append_in_txt(gen_list,gen_nb,fname):
    """ Append generation with number gen_nb in fname (list of (pts,cost) in generation X). """
    f = open("gen_data/" + fname, "a")
    f.write("> GENERATION {0}\n".format(gen_nb+1))
    for (ax,ay,cost_a) in gen_list:
        f.write("{0};{1};{2}\n".format(ax,ay,cost_a))
    f.write("\n")
    f.close()

def read_in_txt(fname):
    """ Read genetic data from fname and load it. """
    dhist = [] ; gen_data = [[],[],[]]
    f = open("gen_data/" + fname, "r")
    txt_lines = f.read().split("\n")
    for line in txt_lines:
        if len(line)>0:
            if line[0] != "#": # caractere de commentaires
                if (line[0] == ">"): # new generation
                    dhist.append(np.array(gen_data)) ; gen_data = [[],[],[]]
                else:
                    elts = line.split(";")
                    for i in [0,1,2]:
                        gen_data[i].append(float(elts[i]))
    f.close()
    return(dhist[1:])