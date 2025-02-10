"""secaggexample: A Flower with SecAgg+ app."""

import random
import numpy as np

def get_dummy_start():
    return np.ones((1, 1))

def get_random_vector(N=5):
    #return np.random.rand(1,N)
    return np.ones((1, N))*3
