import numpy as np
import random
from time import time
import sys
import pickle

__start = str(int(time()))

class __LogPrint(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)    
    def flush(self):
        for f in self.files:
            f.flush()

def start_logging():
    f = open(__start + '.log', 'w')
    sys.stdout = __LogPrint(sys.stdout, f)

__settings = {
    'encoding_size': 25,
    'encoding_lower': -1,
    'encoding_upper': 1,
    'max_order': 0,
}

def generate(order, seed=42):
    global __settings
    __settings['max_order'] = max(order, __settings['max_order'])

    random.seed(seed)
    symbols = list(range(order + 3))
    random.shuffle(symbols)
    sequences = []
    for i in range(2):
        start = order + i - 1
        sequences.append([symbols[start]] + symbols[:order-1] + [symbols[-1]])
        sequences.append([symbols[start]] + symbols[:order-1][::-1] + [symbols[-2]])
    return sequences


def MAPE(groundTruth, prediction):
    return np.nanmean(np.abs(groundTruth - prediction)) / np.nanmean(np.abs(groundTruth))

__encodings = None

def reset_encoder(seed=0):
    np.random.seed(seed)
    global __encodings
    __encodings = {}

def save_encodings():
    global __encodings
    with open(__start + '.enc', 'wb') as f:
        pickle.dump(__encodings, f)

def read_encodings(fname):
    global __encodings
    with open(fname, 'rb') as f:
        __encodings = pickle.load(f)

def encode(symbol):
    global __encodings, __settings
    if symbol not in __encodings:
        __encodings[symbol] = np.random.uniform(
            __settings['encoding_lower'],
            __settings['encoding_upper'],
            __settings['encoding_size'])
    return __encodings[symbol]

def decode(vector):
    global __encodings
    mind = float('inf')
    mink = None
    for k, v in __encodings.items():
        dist = np.sqrt(np.sum(np.square(v - vector)))
        if dist < mind:
            mind = dist
            mink = k
    return mink

__seen_noise = set()

def noise():
    global __settings, __seen_noise
    lower = __settings['max_order'] + 1
    ret = np.random.randint(lower, lower+50001)
    if ret in __seen_noise:
        return noise()
    __seen_noise.add(ret)
    return ret
