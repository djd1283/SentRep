"""Here we place available models to use with triplet loss for learning sentence representations"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# if we wish to use BERT, it can be found here: https://github.com/huggingface/transformers

class GRUReader(nn.Module):
    def __init__(self, d_hidden, d_vocab):
        super().__init__()
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