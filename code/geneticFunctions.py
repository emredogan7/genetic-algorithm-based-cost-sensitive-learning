import numpy as np
import scipy as sc
import random

def fitness(f1min,accuracy,cost):
    coef1 = 0.6
    coef2 = 0.4
    if (f1min>= 0.7):
        f = coef1*accuracy + coef2*(1-(cost/102.03))
    else:
        f = 0.7*(coef1*accuracy + coef2*(1-(cost/102.03)))
    return f 

def mutation(indices):
    pnt = np.random.randint(len(indices)-1)
    if (indices[pnt] == 0):
        indices[pnt] = 1
    elif (indices[pnt] == 1):
        indices[pnt] = 0

def crossOver(indices1,indices2):
    crossoverPoint = np.random.randint(1,len(indices1)-2)
    offspring1 = [0]*len(indices1)
    offspring2 = [0]*len(indices1)

    offspring1 = indices1[:crossoverPoint] + indices2[crossoverPoint:]
    offspring2 = indices1[crossoverPoint:] + indices2[:crossoverPoint]

    return offspring1, offspring2