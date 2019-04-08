import os
import sys
import torch
import pickle
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import RealInput, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import Hebbian

learning = bool(sys.argv[1]) if len(sys.argv) > 1 else True

try: 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
except:
    pass

def moving_average(a, n=100):
    moving_average = []
    for i in range(len(a)):
        start = max(0, i - n)
        values = a[start:i+1]
        moving_average.append(np.sum(values) / float(len(values)))
    return np.array(moving_average)

class Encoder():
    def __init__(self, e_size=25):
        self.encodings = {}
        self.e_size = e_size

    def encode(self, x):
        if x in self.encodings:
            return self.encodings[x]
        self.encodings[x] = torch.rand(e_size)
        return self.encodings[x]

    def decode(self, v):
        nearest = None
        best = float('inf')
        for x, e in self.encodings.items():
            dist = (torch.sum((v - e).pow(2))).pow(0.5)
            if dist < best:
                best = dist
                nearest = x
        return nearest
    
    def precode(self, stream):
        for i in stream:
            self.encode(i)
            
class KNNClassifier:
    def __init__(self, k=1):
        self.k = k
        self.data = []
    def add_sample(self, sample, label):
        self.data.append((sample, label))
    def classify(self, new_sample):
        distances = [(None, float('inf'))]*self.k
        for sample, label in self.data:
            dist = torch.sum(torch.abs(new_sample-sample))
            for i, (_, best_dist) in enumerate(distances):
                if dist < best_dist:
                    distances[i] = (label, dist)
                    break
        counts = {}
        for p_label, _ in distances:
            if p_label not in counts:
                counts[p_label] = 0
            counts[p_label] += 1
        counts = list(counts.items())
        counts.sort(key=lambda x: x[1])
        return counts[0][0]
    
    
class Prototype(Network):
    def __init__(self, encoder, dt: float = 1.0, lag: int = 10, n_neurons: int = 100, time: int = 100):
        super().__init__(dt=dt)
        self.learning = learning
        self.n_neurons = n_neurons
        self.lag = lag
        self.encoder = encoder
        self.time = time
        
        for i in range(lag):
            self.add_layer(RealInput(n=encoder.e_size, traces=True), name=f'input_{i+1}')
            self.add_layer(LIFNodes(n=self.n_neurons, traces=True), name=f'column_{i+1}')
            self.add_monitor(Monitor(self.layers[f'column_{i+1}'], ['s'], time=self.time), name=f'monitor_{i+1}')
            w = 0.3 * torch.rand(self.encoder.e_size, self.n_neurons)
            self.add_connection(Connection(source=self.layers[f'input_{i+1}'], target=self.layers[f'column_{i+1}'], update_rule=Hebbian, w=w, nu=[0.1, 0.1]), source=f'input_{i+1}', target=f'column_{i+1}')
        
        for i in range(lag):
            for j in range(lag):
                w = -0.2 * torch.rand(self.n_neurons, self.n_neurons)
                self.add_connection(Connection(source=self.layers[f'column_{i+1}'], target=self.layers[f'column_{j+1}'], w=w), source=f'column_{i+1}', target=f'column_{j+1}')      
                
    def run(self, inpts, **kwargs) -> None:
        inpts = {k:self.encoder.encode(v).repeat(self.time, 1) for k, v in inpts.items()}
        super().run(inpts, self.time, **kwargs)

e_size = 25
lag = 10
n_neurons = 100
runtime = 100

stream = pickle.load(open('dataset.pkl', 'rb'))['noisy']
encoder = Encoder(e_size)
encoder.precode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

net = Prototype(encoder, lag=lag, time=runtime, n_neurons=n_neurons)
classifier = KNNClassifier()
correct = []

print('it,target,csnn_prediction,accuracy')
for cur in range(10, len(stream)):
    inpt = {f'input_{i+1}': stream[cur+i-10] for i in range(lag)}
    target = stream[cur]
    net.reset_()
    net.run(inpt)
    
    readout = torch.zeros(lag, runtime*n_neurons)
    for i in range(lag):
        readout[i] = net.monitors[f'monitor_{i+1}'].get('s').view(-1)
    prediction = classifier.classify(readout)
    
    if stream[cur+1] >= 10:
        correct.append(int(prediction == target))
        print(f'{cur},{target},{prediction},{moving_average(correct)[-1]}')
        sys.stdout.flush()
        
    classifier.add_sample(readout, target)