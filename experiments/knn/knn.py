import sys
import torch
import pickle
import numpy as np
from argparse import ArgumentParser
args = None


def moving_average(a, n=100):
    moving = []
    for i in range(len(a)):
        start = max(0, i - n)
        values = a[start:i+1]
        moving.append(np.sum(values) / float(len(values)))
    return np.array(moving)


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


class KNN:
    def __init__(self, k=1):
        self.k = k
        self.data = []

    def add_sample(self, sample, label):
        self.data.append((sample, label))

    def classify(self, new_sample):
        distances = [(None, sys.maxsize)]*self.k
        for sample, label in self.data:
            dist = torch.sum(torch.abs(new_sample-sample))
            for i, (_, best_dist) in enumerate(distances):
                if dist < best_dist:
                    distances[i] = (label, dist)
                    break
        counts = {}
        for p_label, _ in distances:
            if p_label not in counts:
                counts[p_label] = 0
            counts[p_label] += 1
        counts = list(counts.items())
        counts.sort(key=lambda x: x[1])
        return counts[0][0]


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

    model = KNN(args.k)
    correct = []

    print('it,target,knn_prediction,accuracy')
    for cur in range(10, len(stream)-1):
        inpt = stream[cur-10:cur]
        readout = torch.zeros(10*25)
        for j, s in enumerate(inpt):
            readout[j*25:(j+1)*25] = encoder.encode(s)
        target = stream[cur]

        prediction = model.classify(readout)

        if stream[cur+1] >= 10:
            correct.append(int(prediction == target))
            print(f'{cur},{target},{prediction},{moving_average(correct)[-1]}')
            sys.stdout.flush()
        
        model.add_sample(readout, target)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--e_size', type=int, default=25)
    parser.add_argument('--lag', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--noise_level', type=int, default=0)
    parser.add_argument('--clean', action='store_true', default=False)
    args = parser.parse_args()

    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    try:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except AssertionError:
        pass

    main()
