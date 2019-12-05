import senteval
import numpy as np
import logging
import torch
from models import GRUReader
import youtokentome as yttm
from datasets import format_sentence, format_sentence_with_bert
from train import calculate_model_outputs
from transformers import BertModel, BertTokenizer
import os
from config import *

params_senteval = {'task_path': path_to_senteval, 'usepytorch': True, 'kfold': 10, 'batch_size': 64,
                   'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}}


def prepare(params, samples):
    # TODO here we load the trained sentence representation model
    if bert:
        grureader = GRUReader(d_hidden=d_hidden, d_in=768)
    else:
        grureader = GRUReader(d_hidden=d_hidden, d_vocab=d_vocab)
    save_details = torch.load(model_path)
    print('Loading model from save')
    grureader.load_state_dict(save_details['state_dict'])
    bpe = yttm.BPE(model=bpe_path)
    params['device'] = torch.device('cuda:0')
    params['model'] = grureader
    params['model'].to(params['device'])  # send model to GPU
    params['bpe'] = bpe
    params['bert'] = None
    if bert:
        pretrained_weights = 'bert-base-uncased'
        bert_model = BertModel.from_pretrained(pretrained_weights)
        params['bert'] = bert_model
        params['bert'].to(params['device'])  # send bert to GPU
        params['wordpiece'] = BertTokenizer.from_pretrained(pretrained_weights)


def batcher(params, batch):
    """Returns random vector to evaluate random performance."""
    bpe_batch_indices = []
    for sentence in batch:
        sentence = ' '.join(sentence)
        if bert:
            indices = format_sentence_with_bert(sentence, params['wordpiece'], max_len)
        else:
            indices = format_sentence(sentence, params['bpe'], max_len)
        bpe_batch_indices.append(torch.LongTensor(indices))

    bpe_batch_indices = torch.stack(bpe_batch_indices, 0)

    # send to gpu
    bpe_batch_indices = bpe_batch_indices.to(params['device'])
    if bert_max:
        # we use max over BERT embeddings as sentence representation
        with torch.no_grad():
            all_embs, _ = params['bert'](bpe_batch_indices)[-2:]
            all_embs, _ = torch.max(all_embs, 1)  # get maximum value along the time dimension 1
            all_embs = all_embs.cpu().detach().numpy()
    elif bert_mean:
        # we use mean over BERT embeddings as sentence representation
        with torch.no_grad():
            all_embs, _ = params['bert'](bpe_batch_indices)[-2:]
            all_embs = torch.mean(all_embs, 1)  # get maximum value along the time dimension 1
            all_embs = all_embs.cpu().detach().numpy()
    else:
        # we use model to calculate embeddings
        all_embs = calculate_model_outputs(params['model'], bpe_batch_indices)
        all_embs = all_embs.cpu().detach().numpy()

    return all_embs


logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

se = senteval.engine.SE(params_senteval, batcher, prepare)
transfer_tasks = ['STS2016']

# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
#                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
#                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
#                   'Length', 'WordContent', 'Depth', 'TopConstituents',
#                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
#                   'OddManOut', 'CoordinationInversion']

results = se.eval(transfer_tasks)

print(results)

# for key in results:
#     print(key, results[key]['all'])
#     print()

print('Hello world')

