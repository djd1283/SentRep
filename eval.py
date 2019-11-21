import senteval
import numpy as np
import logging
import os

PATH_TO_SENTEVAL = '/home/ddonahue/SentEval/data'

params_senteval = {'task_path': PATH_TO_SENTEVAL, 'usepytorch': True, 'kfold': 10, 'batch_size': 64,
                   'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}}

def prepare(params, samples):
    return

def batcher(params, batch):
    """Returns random vector to evaluate random performance."""
    return np.random.randn(len(batch), 100)

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

se = senteval.engine.SE(params_senteval, batcher, prepare)
transfer_tasks = ['STS12']
results = se.eval(transfer_tasks)

for key in results:
    print(key, results[key]['all'])
    print()

print('Hello world')