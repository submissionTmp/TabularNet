# -*- coding : utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder.transformer import MatrixEncoder, Norm


class RowColPooling(nn.Module):
    def __init__(self, feat_dim, method="mean"):
        super(RowColPooling, self).__init__()
        self.feat_dim = feat_dim
        self.method = method
        self.norm = Norm(2*feat_dim)
        self.output = nn.Linear(2*feat_dim, feat_dim)

        self.act = F.relu

        self._reset_parameters()

    def forward(self, matrix):
        """matrix has size of [row, col, feat_dim]i, is the feaature matrix of ONE table"""
        """
        Mean-pooling are implemented here.
        """
        row_count, col_count, feat_dim = matrix.shape
        assert feat_dim == self.feat_dim, "input's  feature dim not math with"

        if self.method == "mean":
            row = torch.mean(matrix, 1, True).repeat(1, col_count, 1)
            col = torch.mean(matrix, 0, True).repeat(row_count, 1, 1)
        elif self.methos == "max":
            row = torch.max(matrix, 1, True).repeat(1, col_count, 1)
            col = torch.max(matrix, 0, True).repeat(row_count, 1, 1)

        pool = torch.cat((row, col), dim=2)
        pool = self.act(self.output(pool))

        return pool

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
