# -*- coding : utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, feat_dim,  max_seq_len=4000):
        """faet_dim: the dime of cells' feature
           max_seq_len: """
        super(PositionalEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, feat_dim)
        for pos in range(max_seq_len):
            for i in range(0, feat_dim, 2):
                pe[pos, i] = math.sin(pos/(10000**((2*i)/feat_dim)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*(i+1))/feat_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # making embedding relatively larger
        x = x * math.sqrt(self.feat_dim)

        # add constant to embedding
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, feat_dim, dropout):
        super(MultiHeadAttention, self).__init__()

        self.feat_dim = feat_dim
        self.feat_h = feat_dim // heads
        assert self.feat_h * heads == feat_dim, "feat_dim should be divided by heads"
        self.heads = heads

        self.q_linear = nn.Linear(feat_dim, feat_dim)
        self.k_linear = nn.Linear(feat_dim, feat_dim)
        self.v_linear = nn.Linear(feat_dim, feat_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, q, k, v, mask):
        """q,k,v should have the size=(bz, seq_len, feat_dim),
        mask should have the size=(bz, seq_len, seq_len)"""

        bz = q.shape[0]
        seq_len = q.shape[1]

        # linear operation and split into h heads
        q = self.q_linear(q).reshape(bz, seq_len, self.heads, self.feat_h)
        k = self.k_linear(k).reshape(bz, seq_len, self.heads, self.feat_h)
        v = self.v_linear(v).reshape(bz, seq_len, self.heads, self.feat_h)

        # transpose
        q = q.transpose(1, 2).reshape(bz*self.heads, seq_len, self.feat_h)
        k = k.transpose(1, 2).reshape(bz*self.heads, seq_len, self.feat_h)
        v = v.transpose(1, 2).reshape(bz*self.heads, seq_len, self.feat_h)

        atten_out = self.attention(q, k, v, mask)
        atten_out = atten_out.reshape(bz, self.heads, seq_len, self.feat_h)
        atten_out = atten_out.transpose(1, 2).reshape(
            bz, seq_len, self.heads*self.feat_h)

        return self.out(atten_out)

    def attention(self, q, k, v, mask):
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.feat_h)
        mask = mask.repeat(self.heads, 1, 1)
        scores = scores.masked_fill(mask == 0, - 1e-9)
        scores = self.dropout(F.softmax(scores, dim=- 1))

        output = torch.bmm(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, feat_dim, ff_dim, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(feat_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.elu(self.linear1(x)))
        x = self.linear2(x)
        return x


class Norm(nn.Module):
    def __init__(self, feat_dim, eps=1e-6):
        super().__init__()

        self.size = feat_dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / \
            (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# build an encoder layer with one multi-head attention layer and one # feed-forward layer


class EncoderLayer(nn.Module):
    def __init__(self, feat_dim, heads, dropout):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(feat_dim)
        self.norm_2 = Norm(feat_dim)
        self.attn = MultiHeadAttention(heads, feat_dim, dropout)
        self.ff = FeedForward(feat_dim, 500, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MatrixEncoder(nn.Module):
    def __init__(self, feat_dim, NumLayers, heads, dropout):
        super(MatrixEncoder, self).__init__()
        self.NumLayers = NumLayers
        self.pe = PositionalEncoder(feat_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(feat_dim, heads, dropout) for l in range(NumLayers)])
        self.norm = Norm(feat_dim)

    def forward(self, x, mask):
        x = self.pe(x)
        for i in range(self.NumLayers):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, feat_dim, NumLayers, heads, dropout):
        super(Attention, self).__init__()
        self.NumLayers = NumLayers
        self.layers = nn.ModuleList(
            [EncoderLayer(feat_dim, heads, dropout) for l in range(NumLayers)])
        self.norm = Norm(feat_dim)

    def forward(self, x, mask):
        for i in range(self.NumLayers):
            x = self.layers[i](x, mask)
        return self.norm(x)
