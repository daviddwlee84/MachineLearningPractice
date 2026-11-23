import numpy as np

def log_normalize(vector):
    return np.log(vector) - np.log(np.sum(vector))

def log_sum(vector):
    pass
    