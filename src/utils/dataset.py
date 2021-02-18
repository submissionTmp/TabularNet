# -*- coding: utf-8 -*-
import os
import random
import pickle as pkl
import networkx as nx
import torch_geometric as tg
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data


class dataset(Dataset):
    def __init__(self, root, data_list, sim):
        self.root = root
        self.data_list = list(data_list)
        self.sim = sim

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        load_path = os.path.join(self.root, file_name)
        with open(load_path, "rb") as f:
            table = pkl.load(f)
        feat = torch.Tensor(table["table_feat"])
        label = torch.Tensor(table["table_label"]).long()
        index = torch.Tensor(table["table_index"]).long()
        pyg_graph = tg.utils.from_networkx(table["G_edge"])

        graph2 = nx.DiGraph()
        graph2.add_nodes_from(table['G_edge'].nodes())

        if self.sim in table:
            # shape [E,2]-> wnsim [2,E]
            wnsim = torch.LongTensor(table[self.sim]).T
            edges = [(i[0], i[1]) for i in table[self.sim]]
            graph2.add_edges_from(edges)
            graph2 = tg.utils.from_networkx(graph2)
        else:
            wnsim = torch.Tensor([[-1, -1]]).long().T
            graph2 = tg.utils.from_networkx(graph2)
            graph2.edge_index = torch.LongTensor([[], []])
        name = table["name"]
        pyg_graph.x = pyg_graph.x.float()
        return (feat, label, pyg_graph, index, name, graph2)


def collater(batch_list):
    table_feats_list = []
    label_list = []
    pyg_list = []
    index_list = []
    name_list = []
    wnsim_list = []
    for feat, label, pyg, index, name, wnsim in batch_list:
        table_feats_list.append(feat)
        label_list.append(label)
        pyg_list.append(pyg)
        index_list.append(index)
        name_list.append(name)
        wnsim_list.append(wnsim)

    return BatchMatrix(table_feats_list, label_list, pyg_list, index_list, name_list, wnsim_list)


def dataset_split(root, train_size, val_size, seed):
    """
    INPUT:
        -- root: path to dataset.
        -- train/val_size: float
        -- seed: default 42

    OUTPUT: list of file names of each sub-dataset
    """
    data_list = []
    for data in os.listdir(root):
        # table size restrict: less than 10000 cells
        with open(os.path.join(root, data), "rb") as f:
            table = pkl.load(f)
        shape = table["table_label"].shape
        if shape[0] * shape[1] < 10000:
            data_list.append(data)
    random.seed(seed)
    random.shuffle(data_list)
    L = len(data_list)

    idx1 = int(L*train_size)
    idx2 = int(L*(train_size+val_size))

    train_set = data_list[:idx1]
    val_set = data_list[idx1:idx2]
    test_set = data_list[idx2:]

    return train_set, val_set, test_set


def dataloader(root, data_list, batch_size, device,  num_workers=4, sim='wnsim'):
    dataset_obj = dataset(root, data_list, sim)
    return DataLoader(dataset_obj, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collater)


class BatchMatrix(object):
    def __init__(self, table_feats_list, label_list, pyg_list, index_list, name_list, wnsim_list):
        self.table_feats_list = table_feats_list
        self.label_list = label_list
        self.pyg_list = pyg_list
        self.name_list = name_list
        self.wnsim_list = wnsim_list

        self.sizes = [l.shape for l in self.label_list]
        self.merge_index = self._merge_index(index_list)
        self.batch_vol = self.__batch_vol()

        self.row_lens_list = [mat.shape[1] for mat in table_feats_list]
        self.col_lens_list = [mat.shape[0] for mat in table_feats_list]

        self.feat_dim = table_feats_list[0].shape[-1]

    def _merge_index(self, index_list):
        index_flatten = [x.flatten() for x in index_list]
        index_lens = [len(ind) for ind in index_flatten]
        len_cum = [0] + [sum(index_lens[:i])
                         for i in range(1, len(index_lens))]
        index_cum = [x+cum for x, cum in zip(index_flatten, len_cum)]
        return torch.cat(index_cum)

    def __batch_vol(self):
        row_cnt = []
        col_cnt = []
        for table in self.label_list:
            row_cnt.append(table.shape[0])
            col_cnt.append(table.shape[1])
        return max(sum(row_cnt)*max(col_cnt), max(row_cnt)*sum(col_cnt))

    def row_along_cat(self, device):
        max_row_len = max(self.row_lens_list)

        row_mask_list = []
        masked_mat_list = []
        for b, mat in enumerate(self.table_feats_list):
            # pad the matrix
            padding = torch.zeros((mat.shape[0], max_row_len-mat.shape[1],
                                   self.feat_dim))
            masked_mat_list.append(torch.cat((mat, padding), dim=1))
            # get the mask
            mask = torch.ones((mat.shape[0], max_row_len, max_row_len))
            mask[:, mat.shape[1]:, :] = 0
            mask[:, :, mat.shape[1]:] = 0
            row_mask_list.append(mask)

        row_mask = torch.cat(row_mask_list, dim=0).to(device)
        masked_mat = torch.cat(masked_mat_list, dim=0).to(device)

        return masked_mat, row_mask

    def col_along_cat(self, device):
        max_col_len = max(self.col_lens_list)

        col_mask_list = []
        masked_mat_list = []
        for b, mat in enumerate(self.table_feats_list):
            # pad the matrix
            padding = torch.zeros((max_col_len-mat.shape[0], mat.shape[1],
                                   self.feat_dim))
            masked_mat_list.append(torch.cat((mat, padding), dim=0))
            # get the mask
            mask = torch.ones((max_col_len, mat.shape[1], max_col_len))
            mask[mat.shape[0]:, :, :] = 0
            mask[:, :, mat.shape[0]:] = 0
            col_mask_list.append(mask)

        col_mask = torch.cat(col_mask_list, dim=1).transpose(0, 1).to(device)
        masked_mat = torch.cat(
            masked_mat_list, dim=1).transpose(0, 1).to(device)

        return masked_mat, col_mask

    def pyg_batch(self, device):
        batch = tg.data.Batch.from_data_list(self.pyg_list)
        return batch.to(device)

    def wn_batch(self, device):
        batch = tg.data.Batch.from_data_list(self.wnsim_list)
        return batch.to(device)

    def row_col_label(self, device):
        """Other=0,
           LeftHeader=1,
           TopHeader=2,
           DataRegion=3"""

        # row label : a row is top header if at least one top-header cell in it
        # col label : a col is left header if at least one left-header cell in it
        row_label = []
        col_label = []
        for label_mat in self.label_list:
            for row in label_mat:
                if 2 in row:
                    row_label.append(1)
                else:
                    row_label.append(0)
            for col in label_mat.T:
                if 1 in col:
                    col_label.append(1)
                else:
                    col_label.append(0)

        row_label = torch.Tensor(row_label).long().to(device)
        col_label = torch.Tensor(col_label).long().to(device)

        return row_label, col_label
