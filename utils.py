# **********************************
#   Author: Suraj Subramanian
#   2nd January 2020
# **********************************

import os, pickle, torch, time, random



def pkl_dump(obj, f):
    with open(f, 'wb') as fo:
        pickle.dump(obj, fo)

        
def pkl_load(f):
    with open(f, 'rb') as fi:
        return pickle.load(fi)
    
def try_gpu():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_filepath(*args):
    fp = os.path.join(*args)
    if os.path.exists(fp):
        ts = ''.join(['_']+random.sample(str(int(time.time())), 5))
        args = list(args)
        args[-1] = args[-1]+ts
        fp = os.path.join(*args)
    else:
        dire = os.path.dirname(fp)
        if not os.path.exists(dire):
            os.makedirs(dire)
    return fp

    
    
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





    


