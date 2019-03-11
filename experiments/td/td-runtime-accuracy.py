import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from dataset import dataset_a, dataset_b, Encoder
from util import start_logging
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.cuda.set_device(1)

class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.fc1 = nn.Linear(10*25, 200)
        self.fc2 = nn.Linear(200, 25)


    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

seed = 100
def refresh(sequence, target, it):
    dataset = dataset_a if it < 10000 else dataset_b
    torch.manual_seed(seed+it)

    c = dataset[torch.randint(0, len(dataset), (1, ))]
    t = c[1:] + [-1] + [-1]
    sequence.extend(c + [int(encoder.noise(it))])
    target.extend(t)
    
criterion = nn.MSELoss()
tdnn = TDNN()
tdnn_optim = optim.SGD(tdnn.parameters(), lr=0.01)
encoder = Encoder(seed)

runtimes = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
history, sequence, target = [], [], []
refresh(sequence, target, 0)
#start_logging()
tdsnn_prediction_label = ''
for runtime in runtimes:
    tdsnn_prediction_label = tdsnn_prediction_label + f',tdsnn_{runtime}ms_prediction'
print(f'it,current,target,random_prediction,tdnn_prediction{tdsnn_prediction_label},tdnn_tl')
for it in range(0, 20000):
    csymbol = sequence.pop(0)
    tsymbol = target.pop(0)
    history.append(csymbol)

    if len(sequence) == 0:
        refresh(sequence, target, it)

    if it < 11:
        continue
        
    tdnn_tl = 0
        
    if it > 1000:
        train = history[max(0, len(history) - 1000):]
        for _ in range(1):
            for i in range(len(train)-11):
                x = train[i:i+10]
                y = train[i+10]
                
                x_enc = torch.zeros(10*25)
                for j, s in enumerate(x):
                    x_enc[j*25:(j+1)*25] = encoder.encode(s)
                y_enc = encoder.encode(train[i+10])
                    
                tdnn_optim.zero_grad()
                tdnn_y_hat = tdnn(x_enc)
                tdnn_loss = criterion(tdnn_y_hat, y_enc)
                tdnn_loss.backward()
                tdnn_optim.step()
                tdnn_tl += tdnn_loss.item()
    
    if len(sequence) == 2:
        x = history[-10:]
        x_enc = torch.zeros(10*25)
        for j, s in enumerate(x):
            x_enc[j*25:(j+1)*25] = encoder.encode(s)
            
        snn = ann_to_snn(tdnn, input_shape=(1, 250))
        snn.add_monitor(monitor=Monitor(obj=snn.layers['2'], state_vars=['s']), name='output_monitor')
        snn.run({'Input': x_enc.repeat(max(runtimes), 1)}, time=max(runtimes))
        
        tdnn_output = tdnn(x_enc)
        tdnn_prediction = encoder.decode(tdnn_output)
        
        output_spikes = snn.monitors['output_monitor'].get('s')
        tdsnn_prediction = ''
        
        for runtime in runtimes:
            spikes = output_spikes[:, :runtime]
            output = torch.sum(spikes, dim=1).float() / (runtime*0.5)
            prediction = encoder.decode(output)
            tdsnn_prediction = tdsnn_prediction + f',{prediction}'
            
        print(f'{it},{csymbol},{tsymbol},{tdnn_prediction}{tdsnn_prediction},{tdnn_tl}')
        sys.stdout.flush()