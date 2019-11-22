import senteval
import numpy as np
import logging
import torch
from models import GRUReader
import youtokentome as yttm
from datasets import format_sentence
import os

PATH_TO_SENTEVAL = '/home/ddonahue/SentEval/data'

params_senteval = {'task_path': PATH_TO_SENTEVAL, 'usepytorch': True, 'kfold': 10, 'batch_size': 64,
                   'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}}

model_path = 'data/grureader.ckpt'
bpe_path = 'data/bpe.model'
d_hidden = 300
d_vocab = 10000
max_len = 40

def prepare(params, samples):
    # TODO here we load the trained sentence representation model
    grureader = GRUReader(d_hidden, d_vocab)
    save_details = torch.load(model_path)
    print('Loading model from save')
    grureader.load_state_dict(save_details['state_dict'])
    bpe = yttm.BPE(model=bpe_path)
    params['device'] = torch.device('cuda:0')
    params['model'] = grureader
    params['bpe'] = bpe


def batcher(params, batch):
    """Returns random vector to evaluate random performance."""
    bpe_batch_indices = []
    for sentence in batch:
        sentence = ' '.join(sentence)
        indices = format_sentence(sentence, params['bpe'], max_len)
        bpe_batch_indices.append(torch.LongTensor(indices))

    bpe_batch_indices = torch.stack(bpe_batch_indices, 0)

    # send to gpu
    bpe_batch_indices = bpe_batch_indices.to(params['device'])
    params['model'].to(params['device'])

    batch_embs = params['model'](bpe_batch_indices).cpu().detach().numpy()

    return batch_embs

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

se = senteval.engine.SE(params_senteval, batcher, prepare)
transfer_tasks = ['SST2']

# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
#                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
#                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
#                   'Length', 'WordContent', 'Depth', 'TopConstituents',
#                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
#                   'OddManOut', 'CoordinationInversion']

results = se.eval(transfer_tasks)

# for key in results:
#     print(key, results[key]['all'])
#     print()

print('Hello world')