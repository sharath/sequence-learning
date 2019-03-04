import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from dataset import dataset_a, dataset_b, Encoder
torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
tdnn_optim = optim.SGD(tdnn.parameters(), lr=0.01, momentum=0.1)
tdpae_optim = optim.SGD(tdpae.parameters(), lr=0.01, momentum=0.1)
encoder = Encoder(seed)

runtime = 500
history, sequence, target = [], [], []
refresh(sequence, target, 0)

print('it,current,target,random_prediction,tdnn_prediction,tdpae_prediction,tdsnn_prediction,tdnn_tl,tdpae_tl')
for it in range(0, 20000):
    csymbol = sequence.pop(0)
    tsymbol = target.pop(0)
    history.append(csymbol)

    if len(sequence) == 0:
        refresh(sequence, target, it)

    if it < 11:
        continue
        
    tdpae_tl = 0
    tdnn_tl = 0
        
    if it > 1000:
        train = history[max(0, len(history) - 3000):]
        for _ in range(5):
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
                tdpae_tl += tdpae_loss.item()
                
                tdnn_optim.zero_grad()
                tdnn_y_hat = tdnn(x_enc)
                tdnn_loss = criterion(tdnn_y_hat, y_enc[225:])
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
        snn.run({'Input': x_enc.repeat(runtime, 1)}, time=runtime)
            
        tdpae_output = tdpae(x_enc)
        tdpae_prediction = encoder.decode(tdpae_output[225:])
        
        tdnn_output = tdnn(x_enc)
        tdnn_prediction = encoder.decode(tdnn_output)
        
        output_spikes = snn.monitors['output_monitor'].get('s')
        tdsnn_output = torch.sum(output_spikes, dim=1).float() / (runtime*0.5)
        tdsnn_prediction = encoder.decode(tdsnn_output)
        
        random_prediction = encoder.decode(torch.rand((25,))*2 - 1)
    
        print(f'{it},{csymbol},{tsymbol},{random_prediction},{tdnn_prediction},{tdpae_prediction},{tdsnn_prediction},{tdnn_tl},{tdpae_tl}')