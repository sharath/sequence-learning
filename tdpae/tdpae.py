import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
from util import start_logging
from dataset import dataset_a, dataset_b, Encoder

class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(10*25, 200),
            nn.Linear(200, 25)
        )

    def forward(self, x):
        return self.seq(x)
    
class TDPAE(nn.Module):
    def __init__(self):
        super(TDPAE, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(10*25, 200),
            nn.Linear(200, 10*25)
        )

    def forward(self, x):
        return self.seq(x)
    
seed = 10
def refresh(sequence, target, it):
    dataset = dataset_a if it < 10000 else dataset_b
    torch.manual_seed(seed+it)

    c = dataset[torch.randint(0, len(dataset), (1, ))]
    t = c[1:] + [-1] + [-1]
    sequence.extend(c + [int(encoder.noise(it))])
    target.extend(t)
    

criterion = nn.MSELoss()
tdnn = TDNN()
tdpae = TDPAE()
tdnn_optim = optim.SGD(tdnn.parameters(), lr=0.01)#, momentum=0.7)
tdpae_optim = optim.SGD(tdpae.parameters(), lr=0.01)#, momentum=0.7)
encoder = Encoder(seed)

history, sequence, target = [], [], []
refresh(sequence, target, 0)
start_logging()
print('it,current,target,tdnn_prediction,tdpae_prediction')
for it in range(0, 20000):
    csymbol = sequence.pop(0)
    tsymbol = target.pop(0)
    history.append(csymbol)

    if len(sequence) == 0:
        refresh(sequence, target, it)

    if it < 11:
        continue
        
    if it > 1000:
        train = history[max(0, len(history) - 1000):]

        for i in range(len(train)-10):
            x = train[i:i+10]
            y = train[i+1:i+11]
            
            x_enc = torch.zeros(10*25)
            for j, s in enumerate(x):
                x_enc[j*25:(j+1)*25] = encoder.encode(s)
                
            y_enc = torch.zeros(10*25)
            for j, s in enumerate(y):
                y_enc[j*25:(j+1)*25] = encoder.encode(s)
                
            tdpae_optim.zero_grad()
            tdpae_y_hat = tdpae(x_enc)
            tdpae_loss = criterion(tdpae_y_hat, y_enc)
            tdpae_loss.backward()
            tdpae_optim.step()
            
            tdnn_optim.zero_grad()
            tdnn_y_hat = tdnn(x_enc)
            tdnn_loss = criterion(tdnn_y_hat, y_enc[225:])
            tdnn_loss.backward()
            tdnn_optim.step()


    x = history[-10:]
    x_enc = torch.zeros(10*25)
    for j, s in enumerate(x):
        x_enc[j*25:(j+1)*25] = encoder.encode(s)
        
    tdpae_output = tdpae(x_enc)
    tdpae_prediction = encoder.decode(tdpae_output[225:])
    
    tdnn_output = tdnn(x_enc)
    tdnn_prediction = encoder.decode(tdnn_output)
    
    
    if len(sequence) == 2:
        print(f'{it},{csymbol},{tsymbol},{tdnn_prediction},{tdpae_prediction}')
        sys.stdout.flush()