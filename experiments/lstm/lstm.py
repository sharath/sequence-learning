from argparse import ArgumentParser
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


args = None


class LSTMOnline(nn.Module):
    def __init__(self):
        super(LSTMOnline, self).__init__()
        self.fc1 = nn.Linear(25, 20)
        self.lstm = nn.LSTM(20, 20)
        self.fc2 = nn.Linear(20, 25)
        self.hidden = (torch.zeros(1, 1, 20), torch.zeros(1, 1, 20))

    def forward(self, x):
        x = self.fc1(x)
        x, self.hidden = self.lstm(x.view(1, 1, -1), self.hidden)
        x = self.fc2(x)
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
        u = torch.rand(1)*100
        if u < args.noise_level:
            ret[i] = subs[int(torch.randint(0, len(subs), (1, )))]
    return ret


def moving_average(a, n=100):
    moving_average = []
    for i in range(len(a)):
        start = max(0, i - n)
        values = a[start:i+1]
        moving_average.append(np.sum(values) / float(len(values)))
    return np.array(moving_average)


def main(args):
    lstm = LSTMOnline()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(lstm.parameters(), lr=0.01)
    original_stream = pickle.load(open('dataset.pkl', 'rb'))['clean' if args.clean else 'noisy']

    encoder = Encoder()
    encoder.precode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    stream = add_noise(original_stream)

    lstm_accuracy = []
    print('it,lstm_accuracy,lstm_loss')
    for it in range(11, 20000):
        lstm.hidden = (torch.zeros(1, 1, 20), torch.zeros(1, 1, 20))
        for i in range(max(0, it-1000), it-1):
            x = stream[i]
            y = stream[i+1]

            x_enc = encoder.encode(x)
            y_enc = encoder.encode(y)

            optimizer.zero_grad()
            y_hat = lstm(x_enc)
            loss = criterion(y_hat, y_enc)
            loss.backward(retain_graph=True)
            optimizer.step()

        if original_stream[it+2] > 10 or original_stream[it+2] == -1:
            x = stream[it]
            y = original_stream[it+1]

            x_enc = encoder.encode(x)
            y_enc = encoder.encode(y)

            lstm.hidden = (torch.zeros(1, 1, 20), torch.zeros(1, 1, 20))
            lstm_output = None
            with torch.no_grad():
                for i in range(max(0, it-1000), it):
                    x_enc = encoder.encode(stream[i])
                    lstm_output = lstm(x_enc)

            lstm_prediction = encoder.decode(lstm_output)
            lstm_loss = torch.pow(torch.sum(torch.pow(y_enc - lstm_output, 2)), 0.5)

            lstm_accuracy.append(float(y == lstm_prediction))

            lstm_moving = np.mean(lstm_accuracy[max(0, len(lstm_accuracy) - 101):len(lstm_accuracy)])

            print(f'{it},{lstm_moving},{lstm_loss}')
            sys.stdout.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise_level', type=int, default=0)
    parser.add_argument('--clean', action='store_true', default=False)
    args = parser.parse_args()

    print(args)

    torch.manual_seed(args.seed)
    try:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except:
        pass

    main(args)

