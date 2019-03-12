import sys
import torch
import torch.nn as nn
import torch.optim as optim

from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from dataset import dataset_a, dataset_b, Encoder
from argparse import ArgumentParser

class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.fc1 = nn.Linear(10*25, 200)
        self.fc2 = nn.Linear(200, 25)


    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def refresh(encoder, sequence, target, it, args):
    torch.manual_seed(args.seed+it)
    dataset = dataset_a if it < 10000 else dataset_b

    c = list(dataset[torch.randint(0, len(dataset), (1, ))])
    for i in range(len(c)):
        u = float(torch.rand(1))
        if u < args.noise:
            c[i] = encoder.noise()
    t = c[1:] + [-1] + [-1]

    sequence.extend(c + [int(encoder.noise())])
    target.extend(t)

def main(args):
    criterion = nn.MSELoss()
    tdnn = TDNN()
    tdnn_optim = optim.SGD(tdnn.parameters(), lr=0.01)
    encoder = Encoder()

    history, sequence, target = [], [], []
    refresh(encoder, sequence, target, 0, args)
    print('it,current,target,tdnn_prediction,tdsnn_prediction,tdnn_tl')

    for it in range(0, 20000):
        csymbol = sequence.pop(0)
        tsymbol = target.pop(0)
        history.append(csymbol)

        if len(sequence) == 0:
            refresh(encoder, sequence, target, it, args)

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
            snn.run({'Input': x_enc.repeat(args.runtime, 1)}, time=args.runtime)
        
            tdnn_output = tdnn(x_enc)
            tdnn_prediction = encoder.decode(tdnn_output)
            output_spikes = snn.monitors['output_monitor'].get('s')
        
            tdsnn_output = (torch.sum(output_spikes, dim=1).float() / (args.runtime*0.5))
            tdsnn_prediction = encoder.decode(tdsnn_output)
    
            print(f'{it},{csymbol},{tsymbol},{tdnn_prediction},{tdsnn_prediction},{tdnn_tl}')
            sys.stdout.flush()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--seed', type=int, default=0, help='psuedorandom seed (default: 0)')
    argparser.add_argument('--noise', type=float, default=0, help='percentage of noise in datastream (default: 0, allowed: 0-1)')
    argparser.add_argument('--runtime', type=int, default=500, help='amount of time to simulate snn (default: 500)')
    argparser.add_argument('--cpu', action='store_true', default=False, help='whether to use the cpu (default: False)')
    args = argparser.parse_args()
    
    torch.manual_seed(args.seed)
    if not args.cpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(args)
    main(args)