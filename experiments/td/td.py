from argparse import ArgumentParser
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bindsnet.conversion import SubtractiveResetIFNodes, ann_to_snn
from bindsnet.network.monitors import Monitor


args = None


class TDNN(nn.Module):
    def __init__(self):
        super(TDNN, self).__init__()
        self.fc1 = nn.Linear(10*25, 200)
        self.fc2 = nn.Linear(200, 25)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Encoder:
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
        u = torch.rand(1)*100
        if u < args.noise_level:
            ret[i] = subs[int(torch.randint(0, len(subs), (1, )))]
    return ret


def moving_average(a, n=100):
    moving = []
    for i in range(len(a)):
        start = max(0, i - n)
        values = a[start:i+1]
        moving.append(np.sum(values) / float(len(values)))
    return np.array(moving)


def convert(ann, dataset):
    snn = ann_to_snn(ann, input_shape=(1, 10*25), data=dataset)
    if not args.no_negative:
        nr = SubtractiveResetIFNodes(
            50, [50], False, refrac=0, reset=0, thresh=1)
        nr.dt = 1.0
        nr.network = snn
        snn.layers['2'] = nr
        nw = torch.zeros((snn.connections[('1', '2')].w.shape[0], snn.connections[('1', '2')].w.shape[1]*2))
        nw[:, :25] = snn.connections[('1', '2')].w
        nw[:, 25:] = -snn.connections[('1', '2')].w
        b = torch.zeros(50)
        b[:25] = snn.connections[('1', '2')].b
        b[25:] = -snn.connections[('1', '2')].b
        snn.connections[('1', '2')].w = nw
        snn.connections[('1', '2')].b = b
        snn.connections[('1', '2')].target = snn.layers['2']
    snn.add_monitor(monitor=Monitor(obj=snn.layers['2'], state_vars=['s']), name='output_monitor')
    return snn


def snn_output(snn):
    if not args.no_negative:
        output_spikes = torch.sum(snn.monitors['output_monitor'].get('s'), dim=1).float()
        subtracted_output_spikes = output_spikes[:25] - output_spikes[25:]
        tdsnn_output = subtracted_output_spikes / args.runtime
        return tdsnn_output
    output_spikes = snn.monitors['output_monitor'].get('s')
    tdsnn_output = torch.sum(output_spikes, dim=1).float() / args.runtime
    return tdsnn_output


def main():
    original_stream = pickle.load(open('dataset.pkl', 'rb'))['clean' if args.clean else 'noisy']
    encoder = Encoder()
    encoder.precode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    stream = add_noise(original_stream)

    tdnn = TDNN()
    tdsnn = convert(tdnn, dataset=None)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(tdnn.parameters(), lr=args.nu, momentum=args.momentum)

    tdnn_accuracy = []
    tdsnn_accuracy = []

    print('it,tdnn_accuracy,tdsnn_accuracy,tdnn_loss,tdsnn_loss,conversion_loss')
    for it in range(11, 20000):
        if it > 0 and it % 1000 == 0:
            dataset = torch.zeros((2989, 250))
            for _ in range(5):
                for i in range(max(0, it-3000), it-11):
                    x = stream[i:i+10]
                    y = stream[i+10]

                    x_enc = torch.zeros(10*25)
                    for j, s in enumerate(x):
                        x_enc[j*25:(j+1)*25] = encoder.encode(s)
                        dataset[i - max(0, it-3000)][j*25:(j+1)
                                                     * 25] = x_enc[j*25:(j+1)*25]
                    y_enc = encoder.encode(y)

                    y_hat = tdnn(x_enc)
                    optimizer.zero_grad()
                    loss = criterion(y_hat, y_enc)
                    loss.backward()
                    optimizer.step()
            tdsnn = convert(tdnn, dataset[:100])

        if original_stream[it+2] > 10 or original_stream[it+2] == -1:
            x = stream[it-9:it+1]
            y = original_stream[it+1]

            x_enc = torch.zeros(10*25)
            for j, s in enumerate(x):
                x_enc[j*25:(j+1)*25] = encoder.encode(s)
            y_enc = encoder.encode(y)

            tdnn_output = tdnn(x_enc)
            tdnn_prediction = encoder.decode(tdnn_output)
            tdnn_loss = torch.pow(torch.sum(torch.pow(y_enc - tdnn_output, 2)), 0.5)
            tdsnn.reset_()
            tdsnn.run({'Input': x_enc.repeat(args.runtime, 1)}, time=args.runtime)

            tdsnn_output = snn_output(tdsnn)
            tdsnn_prediction = encoder.decode(tdsnn_output)
            tdsnn_loss = torch.pow(torch.sum(torch.pow(y_enc - tdsnn_output, 2)), 0.5)

            tdnn_accuracy.append(float(y == tdnn_prediction))
            tdsnn_accuracy.append(float(y == tdsnn_prediction))
            conversion_loss = torch.pow(torch.sum(torch.pow(tdnn_output - tdsnn_output, 2)), 0.5)

            tdnn_moving = np.mean(tdnn_accuracy[max(0, len(tdnn_accuracy) - 101):len(tdnn_accuracy)])
            tdsnn_moving = np.mean(tdsnn_accuracy[max(0, len(tdsnn_accuracy) - 101):len(tdsnn_accuracy)])

            print(f'{it},{tdnn_moving},{tdsnn_moving},{tdnn_loss},{tdsnn_loss},{conversion_loss}')
            sys.stdout.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--e_size', type=int, default=25)
    parser.add_argument('--noise_level', type=int, default=0)
    parser.add_argument('--no_negative', action='store_true', default=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--runtime', type=int, default=500)
    parser.add_argument('--nu', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0)
    args = parser.parse_args()

    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    try:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except AssertionError:
        pass

    main()
