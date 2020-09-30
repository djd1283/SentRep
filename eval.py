import senteval
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader

from models import GRUReader
import youtokentome as yttm
from datasets import format_sentence, format_sentence_with_bert
from sentence_transformers import SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
from utils import select_model, calculate_model_outputs
from transformers import BertModel, BertTokenizer
import os
import wandb
from config import *
from tgalert import TelegramAlert


def prepare(params, samples):
    # TODO here we load the trained sentence representation model
    params['device'] = torch.device('cuda:0')
    if not opt.bert_max and not opt.bert_mean:
        model = select_model(opt.model_name)
        if not opt.no_restore_eval:
            # no_restore_eval defaults to False, in that case we restore model from save file
            # may want to set that to false to test out untrained models like BERT mean emb
            print('Restoring model from save for SentEval evaluation')
            save_details = torch.load(opt.model_path)
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

    # if opt.bert_max:
    #     # we use max over BERT embeddings as sentence representation
    #     raise NotImplementedError('Max needs to fix padding')
    #     # with torch.no_grad():
    #     #     all_embs, _ = params['bert'](bpe_batch_indices)[-2:]
    #     #     all_embs, _ = torch.max(all_embs, 1)  # get maximum value along the time dimension 1
    #     #     all_embs = all_embs.cpu().detach().numpy()
    # elif opt.bert_mean:
    #     # we use mean over BERT embeddings as sentence representation
    #     with torch.no_grad():
    #         all_embs, _ = params['bert'](bpe_batch_indices)[-2:]
    #         # shape [batch_size, n_tokens, 1]
    #         pad_mask = (bpe_batch_indices != 0).float().unsqueeze(2)
    #
    #         all_embs = (all_embs * pad_mask).sum(1) / pad_mask.sum(1)
    #         # all_embs = torch.mean(all_embs, 1)  # get maximum value along the time dimension 1
    # #         all_embs = all_embs.cpu().detach().numpy()
    # else:
    # we use model to calculate embeddings
    all_embs = calculate_model_outputs(params['model'], bpe_batch_indices)
    all_embs = all_embs.cpu().detach().numpy()

    return all_embs


def evaluate():
    wandb.init(project='sent_repr', config=opt, allow_val_change=True)
    alert = TelegramAlert()
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    # model = select_model(opt.model_name)
    #
    # batch_size = 16
    # sts_reader = STSDataReader('data/stsbenchmark')
    # test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
    # test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    # evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
    #
    # if opt.restore_eval:
    #     print('Loading model from save')
    #     save_details = torch.load(opt.model_path)
    #     model.load_state_dict(save_details['state_dict'])
    #     lowest_loss = save_details['best_loss']
    #     start_epoch = save_details['epoch']
    #     print('Lowest loss: %s' % lowest_loss)
    #     print('Current epoch: %s' % start_epoch)

    se = senteval.engine.SE(opt.params_senteval, batcher, prepare)
    transfer_tasks = ['SST2']

    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']

    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']

    results = se.eval(transfer_tasks)
    print(results)

    accuracies = {}
    for task in results:
        if 'acc' in results[task]:
            accuracies[task] = results[task]['acc']

    accuracies['avg'] = np.mean([accuracies[task] for task in accuracies])

    accuracies['mode'] = 'eval'  # add eval flag to let wandb know this is evaluation and not training
    wandb.config.update(accuracies)

    alert.write('Evaluation complete')


if __name__ == '__main__':

    evaluate()

