import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor

seed = 0
noise_level = float(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(seed)
try: 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
except:
    pass

class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.fc1 = nn.Linear(10*25, 200)
        self.fc2 = nn.Linear(200, 25)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x 

class Encoder():
    def __init__(self):
        self.encodings = {}

    def encode(self, x):
        if x in self.encodings:
            return self.encodings[x]
        self.encodings[x] = torch.rand(25)*2 - 1
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

def add_noise(stream):
    ret = list(stream)
    subs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(len(ret)):
        u = torch.rand(1)
        if u < noise_level:
            ret[i] = subs[int(torch.randint(0, len(subs), (1, )))]
    return ret

tdnn = TDNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(tdnn.parameters(), lr=0.01)
original_stream = pickle.load(open('dataset.pkl', 'rb'))['noisy']
encoder = Encoder()
encoder.precode(original_stream)
stream = add_noise(original_stream)

runtime = 500
print('it,target,tdnn_prediction,tdsnn_prediction,noise_level,training_loss,tdnn_loss,tdsnn_loss,conversion_loss')
for it in range(11, 20000):
    training_loss = 0
    if it > 1000:
        for epoch in range(1):
            for i in range(0, it-11):
                x = stream[i:i+10]
                y = stream[i+10]
            
                x_enc = torch.zeros(10*25)
                for j, s in enumerate(x):
                    x_enc[j*25:(j+1)*25] = encoder.encode(s)
                y_enc = 0.5*encoder.encode(y)+0.5

                y_hat = tdnn(x_enc)
                optimizer.zero_grad()
                loss = criterion(y_hat, y_enc)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

    if original_stream[it+2] > 10 or original_stream[it+2] == -1:
        x = stream[it-9:it+1]
        y = original_stream[it+1]
        
        x_enc = torch.zeros(10*25)
        for j, s in enumerate(x):
            x_enc[j*25:(j+1)*25] = encoder.encode(s)
        y_enc = encoder.encode(y)

        snn = ann_to_snn(tdnn, input_shape=(1, 10*25))
        snn.add_monitor(monitor=Monitor(obj=snn.layers['2'], state_vars=['s']), name='output_monitor')
        snn.run({'Input': x_enc.repeat(runtime, 1)}, time=runtime)

        tdnn_output = tdnn(x_enc)
        tdnn_prediction = encoder.decode(2*tdnn_output-1)

        output_spikes = snn.monitors['output_monitor'].get('s')
        tdsnn_output = torch.sum(output_spikes, dim=1).float() / runtime
        tdsnn_prediction = encoder.decode(2*tdsnn_output - 1)

        tdnn_loss = torch.sum(torch.pow((0.5*y_enc+0.5) - tdnn_output, 2)).float()
        tdsnn_loss = torch.sum(torch.pow((0.5*y_enc+0.5) - tdsnn_output, 2)).float()
        conversion_loss = torch.sum(torch.pow(tdnn_output - tdsnn_output, 2)).float()

        print(f'{it},{y},{tdnn_prediction},{tdsnn_prediction},{noise_level},{training_loss},{tdnn_loss},{tdsnn_loss},{conversion_loss}')
        sys.stdout.flush()
