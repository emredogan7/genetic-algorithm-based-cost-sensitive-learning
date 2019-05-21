import numpy as np
import random

def oversampler(data,factor):
    ones_twos = np.zeros((0,22))
    for z in data:
        if (z[21] == 2 or z[21] == 1):
            ones_twos = np.vstack([ones_twos,z])

    newData = data
    for _ in range(factor):
        newData = np.vstack([newData,ones_twos])
        
    random.shuffle(newData)
    return newData
