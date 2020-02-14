# **********************************
#   Author: Suraj Subramanian
#   2nd January 2020
# **********************************

import os, pickle, torch

def pkl_dump(obj, f):
    with open(f, 'wb') as fo:
        pickle.dump(obj, fo)

        
def pkl_load(f):
    with open(f, 'rb') as fi:
        return pickle.load(fi)
    
def try_gpu():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    
class Accumulator(object):
    """Sum a list of numbers over time. [d2l.ai] """

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0] * len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    


