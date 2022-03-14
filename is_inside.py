import numpy as np

def intersect(point,x1,x2):

    return not(point[1] >= max(x1[1],x2[1]) or point[1] < min(x1[1],x2[1])) and point[0] < min(x1[0], x2[0])

def is_inside(point, poly):
    nb_intersect = 0
    for i in range(0, len(poly)) :
        if i==len(poly)-1 :
            if intersect(point, poly[i], poly[0]):
                nb_intersect+=1
        elif intersect(point, poly[i], poly[i+1]):
            nb_intersect+=1
    return not(nb_intersect%2 == 0)
