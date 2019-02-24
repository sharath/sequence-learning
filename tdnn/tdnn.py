import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import dataset_a, dataset_b, Encoder
from util import start_logging


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10*25, 200)
        self.fc2 = nn.Linear(200, 25)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

seed = 100
def refresh(sequence, target, it):
    dataset = dataset_a if it < 10000 else dataset_b
    torch.manual_seed(seed+it)

    c = dataset[torch.randint(0, len(dataset), (1, ))]
    t = c[1:] + [-1] + [-1]
    sequence.extend(c + [int(encoder.noise(it))])
    target.extend(t)

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)#, momentum=0.9)
encoder = Encoder(seed)

history, sequence, target = [], [], []
refresh(sequence, target, 0)


start_logging()
print('it,current,target,prediction,correct')
for it in range(0, 20000):
    csymbol = sequence.pop(0)
    tsymbol = target.pop(0)

    history.append(csymbol)


    if len(sequence) == 0:
        refresh(sequence, target, it)

    if it < 10:
        continue
    tl = [0]
    if it > 1000:
        train = history[max(0, len(history) - 3000):]
        
        for _ in range(40):
            running_loss = 0 
            for i in range(len(train)-10):
                x = train[i:i+10]
                y = train[i+10]

                x_enc = torch.zeros(10*25)
                for j, s in enumerate(x):
                    x_enc[j*25:(j+1)*25] = encoder.encode(s)
                y_enc = encoder.encode(y)

                optimizer.zero_grad()
                y_hat = net(x_enc)
                loss = criterion(y_hat, y_enc)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        tl.append(running_loss)

    x_enc = torch.zeros(10*25)
    for i in range(-10, 0, 1):
        j = i + 10
        print(j)
        x_enc[j*25:(j+1)*25] = encoder.encode(history[it-i])

    print(x_enc)

    output = net(x_enc)
    print(encoder.encodings)
    prediction = encoder.decode(output)
    correct = int(tsymbol == prediction)
    if len(sequence) == 2:
        print(f'{it},{csymbol},{tsymbol},{prediction},{correct},{sum(tl)/len(tl):.4f}')