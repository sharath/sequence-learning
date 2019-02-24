import torch

dataset_a = [[6, 8, 7, 4, 2, 3, 0],
             [1, 8, 7, 4, 2, 3, 5],
             [6, 3, 4, 2, 7, 8, 5],
             [1, 3, 4, 2, 7, 8, 0],
             [0, 9, 7, 8, 5, 3, 4, 1],
             [2, 9, 7, 8, 5, 3, 4, 6],
             [0, 4, 3, 5, 8, 7, 9, 6],
             [2, 4, 3, 5, 8, 7, 9, 1]]

dataset_b = [[6, 8, 7, 4, 2, 3, 5],
             [1, 8, 7, 4, 2, 3, 0],
             [6, 3, 4, 2, 7, 8, 0],
             [1, 3, 4, 2, 7, 8, 5],
             [0, 9, 7, 8, 5, 3, 4, 6],
             [2, 9, 7, 8, 5, 3, 4, 1],
             [0, 4, 3, 5, 8, 7, 9, 1],
             [2, 4, 3, 5, 8, 7, 9, 6]]



class Encoder():
    def __init__(self, seed=0):
        self.seed = seed
        self.encodings = {}
        self.seen = set()
        torch.manual_seed(seed)

    def encode(self, x):
        if x in self.encodings:
            return self.encodings[x]
        self.encodings[x] = torch.rand(25)*2 - 1
        return self.encodings[x]

    def decode(self, v):
        nearest = None
        best = float('inf')
        for x, e in self.encodings.items():
            dist = (torch.sum((v - x).pow(2))).pow(0.5)
            if dist < best:
                best = dist
                nearest = x
        return nearest

    def noise(self, it):
        torch.manual_seed(self.seed + it)
        n = torch.randint(50000, (1, )) + 10
        while n in self.seen:
            n = torch.randint(50000, (1, )) + 10
        self.seen.add(n)
        return n
