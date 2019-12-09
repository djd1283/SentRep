import numpy as np
import torch
from torch import nn as nn


# NOTE: Below classes taken from jadore Transformer repository:
# https://github.com/jadore801120/attention-is-all-you-need-pytorch
# Thanks!
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_out, d_k, d_v, dropout=0.1, d_in=None):
        """

        :param n_head: number of attention heads
        :param d_out: size of output of multi head attention, same as query input q
        :param d_k: dimension of keys during attention
        :param d_v: dimension of values during attention
        :param dropout: regularization constant
        """

        super().__init__()

        d_in = d_in if d_in is not None else d_out
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_out, n_head * d_k)
        self.w_ks = nn.Linear(d_in, n_head * d_k)
        self.w_vs = nn.Linear(d_in, n_head * d_v)
        # TODO change back to original initialization
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_out + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_in + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_in + d_v)))

        # set all biases to zero
        #nn.init.zeros_(self.w_qs.bias)
        #nn.init.zeros_(self.w_ks.bias)
        #nn.init.zeros_(self.w_vs.bias)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_out)

        # TODO undo this
        self.fc = nn.Linear(n_head * d_v, d_out)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.normal_(self.fc.weight, mean=0, std=1 / d_out)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn