# -*- coding : utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


class GINEEncoder(nn.Module):
    def __init__(self, feat_dim, hid_dim,  dropout):
        super(GINEEncoder, self).__init__()
        self.x_encode = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.ReLU())

        self.nn1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim))

        self.nn2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim))

        self.conv1 = GINEConv(self.nn1, train_eps=True)
        self.conv2 = GINEConv(self.nn2, train_eps=True)

        self.dropout = nn.Dropout(dropout)
        self.act = F.relu
        self._reset_parameters()

        self.attr_encode = nn.Embedding(8, hid_dim)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.x_encode(x)
        edge_attr = self.attr_encode(edge_attr.long()).squeeze()

        x = self.act(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index, edge_attr))

        return x
