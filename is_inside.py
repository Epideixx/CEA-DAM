import numpy as np

def intersect(point,x1,x2):

    if point[0] < min(x1[0], x2[0]) :
        return not(point[1] >= max(x1[1],x2[1]) or point[1] < min(x1[1],x2[1]))
    elif point[0] > max(x1[0], x2[0]) :
        return False
    else :
        if point[1] >= max(x1[1],x2[1]) or point[1] < min(x1[1],x2[1]) :
            return False
        if (x2[0] - x1[0]) == 0 :
            return True

        a = (x2[1] - x1[1])/(x2[0] - x1[0])
        b = x2[1] - a*x2[0]

        if a == 0 :
            return True

        return (point[1]-b)/a > point[0]


def is_inside(point, poly):
    nb_intersect = 0
    for i in range(0, len(poly)) :
        if i==len(poly)-1 :
            if intersect(point, poly[i], poly[0]):
                nb_intersect+=1
        elif intersect(point, poly[i], poly[i+1]):
            nb_intersect+=1
    return not(nb_intersect%2 == 0)


def is_inside_bat(x, bat_list):
    """ Checks if point x is inside a bat defined in bat_list or not. """
    is_inside_bat = False ; i = 0
    while not(is_inside_bat) and i<len(bat_list):
        is_inside_bat = is_inside(x, bat_list[i])
        i += 1
    return(is_inside_bat)