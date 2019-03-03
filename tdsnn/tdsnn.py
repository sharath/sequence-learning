import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from dataset import dataset_a, dataset_b, Encoder
from util import start_logging


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(10*25, 200),
            nn.Linear(200, 25)
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

ann = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(ann.parameters(), lr=0.01)
encoder = Encoder(seed)

history, sequence, target = [], [], []
refresh(sequence, target, 0)
runtime = 500
start_logging()
print('it,current,target,prediction,correct')
for it in range(0, 20000):
    csymbol = sequence.pop(0)
    tsymbol = target.pop(0)

    history.append(csymbol)


    if len(sequence) == 0:
        refresh(sequence, target, it)

    if it < 11:
        continue
    tl = [0]

    if it > 1000:
        train = history[max(0, len(history) - 1000):]
        
        for _ in range(1):
            running_loss = 0 
            for i in range(len(train)-11):
                x = train[i:i+10]
                y = train[i+10]

                x_enc = torch.zeros(10*25)
                for j, s in enumerate(x):
                    x_enc[j*25:(j+1)*25] = encoder.encode(s)
                y_enc = encoder.encode(y)

                optimizer.zero_grad()
                y_hat = ann(x_enc)
                loss = criterion(y_hat, y_enc)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        tl.append(running_loss)

    x = history[-10:]
    x_enc = torch.zeros(10*25)
    for j, s in enumerate(x):
        x_enc[j*25:(j+1)*25] = encoder.encode(s)
    x_enc = x_enc.repeat(runtime, 1)

    snn = ann_to_snn(ann, input_shape=(1, 250))
    snn.add_monitor(monitor=Monitor(obj=snn.layers['2'], state_vars=['s']), name='output_monitor')
    snn.run({'Input': x_enc}, time=runtime)

    output_spikes = snn.monitors['output_monitor'].get('s')
    output = torch.sum(output_spikes, dim=1).float() / (runtime*0.5)

    prediction = encoder.decode(output)
    correct = int(tsymbol == prediction)
    if len(sequence) == 2:
        print(f'{it},{csymbol},{tsymbol},{prediction},{correct},{sum(tl)/len(tl):.4f}')
        sys.stdout.flush()
