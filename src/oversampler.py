import numpy as np
import random

def oversampler(data, sampling_factor):
    ones_twos = np.zeros((0,22))
    for instance in data:
        if (instance[21] == 2 or instance[21] == 1):
            ones_twos = np.vstack([ones_twos,instance])
    
    oversampled_data
    oversampled_data = data
    
    for _ in range(sampling_factor):
        oversampled_data = np.vstack([oversampled_data,ones_twos])
        
    random.shuffle(oversampled_data)
    return oversampled_data
