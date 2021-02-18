from src.encoder.rowcolPooling import RowColPooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder.transformer import Norm
from torch_scatter import scatter_mean, scatter_sum


class BaseModel(nn.Module):
    def __init__(self, feat_dim, hid_dim,  num_class, dropout,  device, args):
        super(BaseModel, self).__init__()
        self.num_class = num_class

        self.ori_linear = nn.Linear(feat_dim, hid_dim)

        self.row_linear1 = nn.Linear(feat_dim, hid_dim)
        self.col_linear1 = nn.Linear(feat_dim, hid_dim)

        self.pooling = RowColPooling(feat_dim)
        self.pooling_linear = nn.Linear(feat_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.loss_fn = nn.NLLLoss()
        self.act = F.relu

    def set_classifier(self, in_dim):
        classifier = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, self.num_class),
            nn.LogSoftmax(dim=1))

        return classifier

    def scatter_mean(self, source, index, dim):
        res = scatter_mean(source, index, dim)
        return res[torch.unique(index)]

    def scatter_sum(self, source, index, dim):
        res = scatter_sum(source, index, dim)
        return res[torch.unique(index)]

    def pyg_batch(self, batch):
        return batch.pyg_batch(self.device)

    def base_forward(self, batch):
        _row_x, row_mask = batch.row_along_cat(self.device)
        _col_x, col_mask = batch.col_along_cat(self.device)

        row_lens = batch.row_lens_list
        col_lens = batch.col_lens_list

        row_x = self.dropout(_row_x)
        col_x = self.dropout(_col_x)

        # Linear transfor row/col_x to a handlable dimention
        row_x = self.row_linear1(row_x)
        col_x = self.col_linear1(col_x)

        return row_x, row_mask, col_x, col_mask,  row_lens, col_lens

    def label(self, batch):
        label = self.label_list_flatten(batch.label_list)

        # change 0, 1, 3, 4, 5 to 0, 1, 2, 3, 4
        for i, v in enumerate(label):
            if v >= 2:
                label[i] -= 1

        return label

    def original_feats(self, batch):
        ori_feats = self.mat_list_flatten(batch.table_feats_list)
        ori_feats = self.act(self.ori_linear(ori_feats))
        ori_feats = self.scatter_mean(
            ori_feats, batch.merge_index.to(self.device), dim=0)

        return ori_feats

    def row_col_pooling(self, batch):
        tables = batch.table_feats_list
        tables = [table.to(self.device) for table in tables]
        table_list = [self.pooling(table) for table in tables]

        pooling_feats = self.mat_list_flatten(table_list)
        pooling_feats = self.pooling_linear(pooling_feats)
        pooling_feats = self.act(pooling_feats)
        pooling_feats = self.scatter_mean(
            pooling_feats, batch.merge_index.to(self.device), dim=0)

        return pooling_feats

    def mat_list_flatten(self, mat_list):
        mat_list = [mat.reshape(-1, mat.shape[-1]) for mat in mat_list]
        flatten_mat = torch.cat(mat_list, dim=0)
        return flatten_mat.to(self.device)

    def label_list_flatten(self, label_list):
        label_list = [la.reshape(-1, 1).squeeze() for la in label_list]
        label = torch.cat(label_list, dim=0)
        return label.squeeze().to(self.device)

    def flatten(self, mat_list, label_list):

        mat_list = [mat.reshape(-1, mat.shape[-1]) for mat in mat_list]
        label_list = [la.reshape(-1, 1).squeeze() for la in label_list]

        flatten_emb = torch.cat(mat_list, dim=0)
        flatten_label = torch.cat(label_list, dim=0)

        return flatten_emb, flatten_label

    def RowCol_emb_flatten(self, row_emb, col_emb, row_lens, col_lens, batch):
        row_along_mat_list = list(torch.split(row_emb, col_lens))
        col_along_mat_list = list(torch.split(
            col_emb.transpose(0, 1), row_lens, dim=1))

        for count, (mat, length) in enumerate(zip(row_along_mat_list, row_lens)):
            tmp_mat = mat[:, :length]
            row_along_mat_list[count] = tmp_mat
        for count, (mat, length) in enumerate(zip(col_along_mat_list, col_lens)):
            tmp_mat = mat[:length, :]
            col_along_mat_list[count] = tmp_mat

        # Aggregate row/col embeddings
        # Cat. embs
        agg_mat_list = []
        for rmat, cmat in zip(row_along_mat_list, col_along_mat_list):
            tmp_mat = torch.cat((rmat, cmat), dim=2)
            agg_mat_list.append(tmp_mat)

        agg_feat = self.mat_list_flatten(agg_mat_list)
        agg_feat = self.scatter_mean(
            agg_feat, batch.merge_index.to(self.device), dim=0)

        return agg_feat

    def x_flatten(self, row_x, row_lens, col_lens):
        row_x_mat_list = list(torch.split(row_x, col_lens))
        for count, (mat, length) in enumerate(zip(row_x_mat_list, row_lens)):
            tmp_mat = mat[:, :length]
            row_x_mat_list[count] = tmp_mat
        mat_list = [mat.reshape(-1, mat.shape[-1]) for mat in row_x_mat_list]
        return torch.cat(mat_list, dim=0)

    def output(self, cat_embs, batch):

        pred = self.classifier(cat_embs)

        label = self.label(batch)
        label = self.scatter_mean(
            label, batch.merge_index.to(self.device), dim=0)

        return pred, label
