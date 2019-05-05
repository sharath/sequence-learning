from argparse import ArgumentParser
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

args = None


def moving_average(a, n=100):
    moving = []
    for i in range(len(a)):
        start = max(0, i - n)
        values = a[start:i + 1]
        moving.append(np.sum(values) / float(len(values)))
    return np.array(moving)


class LSTMOnline(nn.Module):
    def __init__(self):
        super(LSTMOnline, self).__init__()
        self.inpt = nn.Linear(args.e_size, args.h_size)
        self.lstm = nn.LSTM(args.h_size, args.h_size)
        self.out = nn.Linear(args.h_size, args.e_size)

    def forward(self, x, h=None):
        x = self.inpt(x.view(1, -1)).unsqueeze(1)
        x, h = self.lstm(x, h)
        x = self.out(x.squeeze(1))
        return x, h


class Encoder():
    def __init__(self, e_size=25):
        self.encodings = {}
        self.e_size = e_size

    def encode(self, x):
        if x in self.encodings:
            return self.encodings[x]
        self.encodings[x] = 2 * torch.rand(self.e_size) - 1
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
        u = torch.rand(1) * 100
        if u < args.noise_level:
            ret[i] = subs[int(torch.randint(0, len(subs), (1,)))]
    return ret


def main():
    original_stream = pickle.load(open('dataset.pkl', 'rb'))['clean' if args.clean else 'noisy']
    encoder = Encoder(e_size=args.e_size)
    encoder.precode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    stream = add_noise(original_stream)

    model = LSTMOnline()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.nu, momentum=args.momentum)

    correct = []
    prev = None
    print('it,target,prediction,accuracy')
    for it in range(len(stream) - 2):
        hidden = prev
        stored = False
        for jt in range(max(0, it - args.window), it):
            x = stream[jt]
            y = stream[jt + 1]

            x_enc = encoder.encode(x)
            y_enc = encoder.encode(y)

            y_pred, hidden = model(x_enc, hidden)

            if not stored and it >= args.window:
                prev = hidden
                stored = True

            optimizer.zero_grad()
            loss = criterion(y_pred.view(-1), y_enc)
            loss.backward(retain_graph=True)
            optimizer.step()

        if stream[it + 2] == -1 or stream[it + 2] >= 10:
            cur = stream[it]
            tar = original_stream[it + 1]

            with torch.no_grad():
                x_enc = encoder.encode(cur).float()
                y_pred, hidden = model(x_enc, hidden)
                y_pred = encoder.decode(y_pred.detach())

                correct.append(int(y_pred == tar))

                print(f'{it},{tar},{y_pred},{moving_average(correct)[-1]}')
                sys.stdout.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--e_size', type=int, default=25)
    parser.add_argument('--h_size', type=int, default=20)
    parser.add_argument('--window', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--noise_level', type=int, default=0)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--nu', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.2)
    args = parser.parse_args()

    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    try:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except AssertionError:
        pass

    main()
