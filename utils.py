"""Helper functions for train and evaluation"""
from models import GRUReader, BERTGRUReader, BERTMeanEmb
from config import *


def select_model(model_name):
    if model_name == 'grureader':
        model = GRUReader(opt.d_hidden, d_vocab=opt.d_vocab)
    elif model_name == 'bertgrureader':
        model = BERTGRUReader(opt.d_hidden, train_bert=opt.train_bert)
    elif model_name == 'bertmean':
        model = BERTMeanEmb()
    else:
        raise ValueError('No model name %s' % model_name)
    return model


def calculate_model_outputs(model, all_s):
    # here we take wordpiece indices and convert them to embeddings with BERT

    # then we change model to take embeddings as input

    all_emb = model(all_s)

    return all_emb