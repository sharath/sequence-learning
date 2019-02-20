import sys
import numpy as np
from collections import deque
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from pybrain.datasets import SequentialDataSet
from util import *

#dataset = generate(6) + generate(7)

dataset_a = [[6, 8, 7, 4, 2, 3, 0],
             [1, 8, 7, 4, 2, 3, 5],
             [6, 3, 4, 2, 7, 8, 5],
             [1, 3, 4, 2, 7, 8, 0],
             [0, 9, 7, 8, 5, 3, 4, 1],
             [2, 9, 7, 8, 5, 3, 4, 6],
             [0, 4, 3, 5, 8, 7, 9, 6],
             [2, 4, 3, 5, 8, 7, 9, 1]]

dataset_b = [[6, 8, 7, 4, 2, 3, 5],
             [1, 8, 7, 4, 2, 3, 0],
             [6, 3, 4, 2, 7, 8, 0],
             [1, 3, 4, 2, 7, 8, 5],
             [0, 9, 7, 8, 5, 3, 4, 6],
             [2, 9, 7, 8, 5, 3, 4, 1],
             [0, 4, 3, 5, 8, 7, 9, 1],
             [2, 4, 3, 5, 8, 7, 9, 6]]

def refresh(seq, tar, it):
    dataset = dataset_a if it < 10000 else dataset_b
    chosen = np.random.choice(dataset) + [noise()]
    target = chosen[1:] + ['None'] + ['None']
    seq.extend(chosen)
    tar.extend(target)

def window(data):
    start = max(0, len(data) - 1000)
    return data[start:]

net = buildNetwork(25, 20, 25, hiddenclass=LSTMLayer,
                   bias=True, outputbias=False, recurrent=True)

history = []

sequence = deque()
target = deque()
compute_counter = 0
refresh(sequence, target, 0)
start_logging()
print 'it,current,target,correct'
for it in range(0, 20000):
    curs = sequence.popleft()
    tars = target.popleft()
    history.append(curs)

    if len(sequence) == 0:
        refresh(sequence, target, it)

    if it > 0 and it % 1000 == 0: # this isn't needed
        compute_counter = 1000

    if compute_counter != 0:  # train
        compute_counter -= 1
        ds = SequentialDataSet(25, 25)
        tsegment = window(history)

        for j in range(1, len(tsegment)):
            ds.addSample(encode(tsegment[j - 1]),
                         encode(tsegment[j]))

        trainer = RPropMinusTrainer(net, dataset=ds)
        if len(tsegment) > 1:
            trainer.trainEpochs(200)

        # get the hidden state right
        net.reset()
        for symbol in tsegment:
             _ = net.activate(encode(symbol))

    output = net.activate(encode(curs))
    prediction = decode(output)
    correct = tars == prediction

    print ','.join([str(t) for t in [it, tars, prediction, correct]])
    sys.stdout.flush()
    save_encodings()
