"""Here we place available models to use with triplet loss for learning sentence representations"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class SNLIClassifierFromModel(nn.Module):
    def __init__(self, model, model_size):
        super().__init__()
        self.model = model
        self.out_layer = nn.Linear(model_size * 2, 3)

    def forward(self, premise, hypothesis):
        """Map a premise and a hypothesis to an entailment prediction using a particular model"""
        premise_emb = self.model(premise)
        hypothesis_emb = self.model(hypothesis)
        all_emb = torch.cat([premise_emb, hypothesis_emb], -1)
        output = self.out_layer(all_emb)
        return output


class GRUReader(nn.Module):
    def __init__(self, d_hidden, d_vocab=None):
        super().__init__()
        self.d_vocab = d_vocab
        self.emb = nn.Embedding(d_vocab, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.first_lin = nn.Linear(d_hidden, d_hidden * 2)
        self.second_lin = nn.Linear(d_hidden * 2, d_hidden)

    def forward(self, x):
        x_emb = self.emb(x)
        states, last = self.gru(x_emb)
        last = last.squeeze(0)

        output = self.second_lin(torch.relu(self.first_lin(last)))

        return output


class BERTGRUReader(nn.Module):
    def __init__(self, d_hidden, d_bert=768, train_bert=False):
        super().__init__()
        self.train_bert = train_bert
        self.d_bert = d_bert
        self.d_hidden = d_hidden
        self.emb = nn.Linear(d_bert, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.first_lin = nn.Linear(d_hidden, d_hidden * 2)
        self.second_lin = nn.Linear(d_hidden * 2, d_hidden)
        pretrained_weights = 'bert-base-uncased'
        self.bert_model = BertModel.from_pretrained(pretrained_weights)

    def forward(self, x):
        if self.train_bert:
            x_emb = self.bert_model(x)[0]
        else:
            with torch.no_grad():
                x_emb = self.bert_model(x)[0]

        # convert output of bert to input of GRU
        if self.d_bert != self.d_hidden:
            x_emb = self.emb(x_emb)
        states, last = self.gru(x_emb)
        last = last.squeeze(0)

        output = self.second_lin(torch.relu(self.first_lin(last)))

        return output


class BERTMeanEmb(nn.Module):
    def __init__(self, d_bert=768):
        super().__init__()
        self.d_bert = d_bert
        pretrained_weights = 'bert-base-uncased'
        self.bert_model = BertModel.from_pretrained(pretrained_weights)

    def forward(self, x):
        x_emb = self.bert_model(x)[0]
        output = x_emb.mean(dim=1)  # take the mean embedding across the time dimension

        return output