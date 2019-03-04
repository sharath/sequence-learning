import os
import dill
import numpy as np
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.data.inference_shifter import InferenceShifter
from config import MODEL_PARAMS

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

seed = 10

seen_noise = set()
def noise(it):
    global seen_noise, seed
    np.random.seed(seed+it)
    ret = np.random.randint(11,50011)
    while ret in seen_noise:
        ret = np.random.randint(11,50011)
    seen_noise.add(ret)
    return ret


def refresh(seq, tar, it):
    dataset = dataset_a if it < 10000 else dataset_b
    chosen = np.random.choice(dataset)
    target = chosen[1:] + [-1] + [-1]
    seq.extend(chosen + [noise(it)])
    tar.extend(target)


def get_model():
    if os.path.isfile('/home/sharathramku/Jupyter/sequence-learning/htm/htm.model'):
        f = open('htm.model', 'rb')
        d = dill.load(f)
        f.close()
        return d['model']
    f = open('htm.model', 'wb')
    htm = ModelFactory.create(MODEL_PARAMS)
    htm.enableInference({"predictedField": "element"})
    d = {'model': htm}
    dill.dump(d, f)
    f.close()
    return get_model()

htm = get_model()

shifter = InferenceShifter()
print(vars(htm))

history = []
sequence = []
target = []

refresh(sequence, target, 0)

for it in range(1, 20000):
    curs = sequence.pop(0)
    tars = target.pop(0)
    history.append(curs)

    if len(sequence) == 0:
        refresh(sequence, target, it)
    
    result = shifter.shift(htm.run({"element": curs}))
    tm = htm._getTPRegion().getSelf()._tfdr
    