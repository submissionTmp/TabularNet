# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import networkx as nx
import pickle as pkl
from utils import Encoder, merged_cells, positional_encode, heur_edges, tlbr_edges
from bert_serving.client import BertClient


def get_neighbors(node, i, j, nodes_mer, shape):
    neighbors = []
    atts = []
    if j+1 < shape[1]:
        nei = nodes_mer[i, j+1]
        if nei != node:
            neighbors.append(nei)
            atts.append([0])
        else:
            re1, re2 = get_neighbors(nei, i, j+1, nodes_mer, shape)
            neighbors.extend(re1)
            atts.extend(re2)

    if i+1 < shape[0]:
        nei = nodes_mer[i+1, j]
        if nei != node:
            neighbors.append(nei)
            atts.append([2])
        else:
            re1, re2 = get_neighbors(nei, i+1, j, nodes_mer, shape)
            neighbors.extend(re1)
            atts.extend(re2)

    if (i+1 < shape[0]) and (j+1 < shape[1]):
        nei = nodes_mer[i+1, j+1]
        if nei != node:
            neighbors.append(nei)
            atts.append([4])
        else:
            re1, re2 = get_neighbors(nei, i+1, j+1, nodes_mer, shape)
            neighbors.extend(re1)
            atts.extend(re2)

    if (i-1 > 0) and (j+1 < shape[1]):
        nei = nodes_mer[i-1, j+1]
        if nei != node:
            neighbors.append(nei)
            atts.append([6])
        else:
            re1, re2 = get_neighbors(nei, i-1, j+1, nodes_mer, shape)
            neighbors.extend(re1)
            atts.extend(re2)

    return neighbors, atts


def build_graph(data):
    shape = data["table_feat"].shape[:2]
    nodes = data["table_index"].reshape(shape)

    _node = []
    _edge = []
    _edge_att = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            current_node = nodes[i, j]
            if current_node not in _node:
                _node.append(current_node)
                neighbors, atts = get_neighbors(
                    current_node, i, j, nodes, shape)

                _edge.extend([[current_node,  n] for n in neighbors])
                _edge_att.extend(atts)

                _edge.extend([[n, current_node] for n in neighbors])
                _edge_att.extend([[a+1 for a in att] for att in atts])

    return _edge, _edge_att


def vector(nei):
    vec = [nei.count(i)/len(nei) for i in [0, 1, 3, 4, 5]]
    return vec


def build_pyg(data):
    N, M = data["table_label"].shape
    G = nx.DiGraph()

    edge, edge_att = build_graph(data)
    G.add_edges_from(edge)
    nx.set_edge_attributes(
        G, {tuple(e): a for e, a in zip(edge, edge_att)}, "edge_attr")

    feats = data["table_feat"].reshape(-1, 820)[list(set(data["table_index"]))]
    label = data["table_label"].flatten()[list(set(data["table_index"]))]
    nx.set_node_attributes(
        G, {n: x for n, x in zip(list(G.nodes()), feats)}, "x")
    nx.set_node_attributes(
        G, {n: x for n, x in zip(list(G.nodes()), label)}, "label")

    return G


def json2matrix(load_path, save_path):
    """Convert json to matrix-form"""

    table_name = load_path.split("/")[-1][:-4]
    table_jsons = []
    sheet_names = []

    with open(load_path) as f:
        for line in f:
            tj = json.loads(line)
            table_jsons.append(tj)
            sheet_names.append(tj["SheetName"])

    if len(sheet_names) == 1:
        tnames = [table_name+"_"+sheet_names[0]]
    else:
        tnames = [table_name+"_"+sheet+"_" +
                  str(i) for i, sheet in enumerate(sheet_names)]

    for table, name in zip(table_jsons, tnames):
        row_count, col_count = int(
            table["RowCount"]), int(table["ColumnCount"])

        cell_atts = []  # feature engineerings
        cell_text = []  # cells' text content
        cell_role = []  # cell type role
        cell_textual = []

        for row in table["CellAttributes"]:
            for cell in row:
                row_idx = cell["RowNumber"] - table["FirstRowNumber"]
                col_idx = cell["ColumnNumber"] - table["FirstColumnNumber"]

                cell_role.append(int(cell["CellRole"]))

                atts, textual = encode.encode(cell)
                cell_atts.append(atts+positional_encode(row_idx,
                                                        col_idx, row_count, col_count))
                cell_textual.append(textual)

                cell_text.append(cell["Text"])

        row_count, col_count = int(
            table["RowCount"]), int(table["ColumnCount"])

        # Copy information from merged leaders to others
        merged_groups, merged_leaders = merged_cells(
            table["MergedRegions"], col_count, row_count)
        cell_index = list(range(len(cell_role)))
        for group, leader in zip(merged_groups, merged_leaders):
            for mm in group:
                cell_atts[mm] = cell_atts[leader]
                cell_text[mm] = cell_text[leader]
                cell_role[mm] = cell_role[leader]
                cell_textual[mm] = cell_textual[leader]
                cell_index[mm] = leader

        # Empty and nontextual cells are set to be [0]*768
        bert_embs = np.zeros((len(cell_text), 768))
        # BERT encoding non-empty and textual cells' text
        text_lens = np.array([len(text.strip()) for text in cell_text])
        non_empty_idx = np.nonzero(text_lens)[0]
        non_textual_idx = np.nonzero(cell_textual)[0]

        # emb_cell_idx = non_empty_idx
        emb_cell_idx = list(set(non_empty_idx) - set(non_textual_idx))
        if len(emb_cell_idx) != 0:
            text_list = [cell_text[idx] for idx in emb_cell_idx]
            text_embs = bc.encode(text_list)
            bert_embs[emb_cell_idx] = text_embs

        # Concate text_emb and cell_atts
        cell_feats = np.hstack((np.array(cell_atts), bert_embs))

        table_matrix = cell_feats.reshape(
            row_count, col_count, cell_feats.shape[1])
        label_matrix = np.array(cell_role).reshape(row_count, col_count)
        index_matrix = np.array(cell_index)

        data = {"table_feat": table_matrix,
                "table_label": label_matrix, "table_index": index_matrix}

        G = build_pyg(data)
        data["G_edge"] = G
        data["name"] = save_path.split("/")[-1]

        with open(os.path.join(save_path, name), "wb") as f:
            pkl.dump(data, f)


if __name__ == "__main__":
    """This script aims to convert tabular data into a 3-dim array.
    i.e. [row, col, craft_feature+BERT_emb]"""

    encode = Encoder()
    bc = BertClient(check_length=False)

    prefix = "../../data"
    path_to_rawData = prefix+"/CellRole"
    path_to_Save = prefix+"/Processed_CellRole"

    files_list = os.listdir(path_to_rawData)
    t0 = time.time()
    for count, file in enumerate(files_list):
        json2matrix(os.path.join(path_to_rawData, file), path_to_Save)
        if count % 100 == 0:
            print("{:>4}/{:<4}, CumTime={:.1f}s".format(count,
                                                        len(files_list), time.time()-t0))
