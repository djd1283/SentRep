import senteval
import numpy as np
import logging
import torch
from models import GRUReader
import youtokentome as yttm
from datasets import format_sentence, format_sentence_with_bert
from utils import select_model, calculate_model_outputs
from transformers import BertModel, BertTokenizer
import os
import wandb
from config import *


def prepare(params, samples):
    # TODO here we load the trained sentence representation model
    params['device'] = torch.device('cuda:0')
    if not opt.bert_max and not opt.bert_mean:
        model = select_model(opt.model_name)
        save_details = torch.load(opt.model_path)
        print('Loading model from save')
        model.load_state_dict(save_details['state_dict'])
        bpe = yttm.BPE(model=opt.bpe_path)

        params['model'] = model
        params['model'].to(params['device'])  # send model to GPU
        params['bpe'] = bpe
    else:
        bert_model = BertModel.from_pretrained(opt.pretrained_weights)
        params['bert'] = bert_model
        params['bert'].to(params['device'])  # send bert to GPU

    if opt.bert:
        params['wordpiece'] = BertTokenizer.from_pretrained(opt.pretrained_weights)


def batcher(params, batch):
    """Returns random vector to evaluate random performance."""
    bpe_batch_indices = []
    for sentence in batch:
        sentence = ' '.join(sentence)
        if opt.bert:
            indices = format_sentence_with_bert(sentence, params['wordpiece'], opt.max_len)
        else:
            indices = format_sentence(sentence, params['bpe'], opt.max_len)
        bpe_batch_indices.append(torch.LongTensor(indices))

    bpe_batch_indices = torch.stack(bpe_batch_indices, 0)

    # send to gpu
    bpe_batch_indices = bpe_batch_indices.to(params['device'])

    if opt.bert_max:
        # we use max over BERT embeddings as sentence representation
        with torch.no_grad():
            all_embs, _ = params['bert'](bpe_batch_indices)[-2:]
            all_embs, _ = torch.max(all_embs, 1)  # get maximum value along the time dimension 1
            all_embs = all_embs.cpu().detach().numpy()
    elif opt.bert_mean:
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


def evaluate():
    wandb.init(project='sent_repr', config=opt, allow_val_change=True)

    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(opt.params_senteval, batcher, prepare)
    # transfer_tasks = ['SST2']

    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']

    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']

    results = se.eval(transfer_tasks)
    print(results)

    wandb.config.update({'eval': True})


if __name__ == '__main__':
    evaluate()

