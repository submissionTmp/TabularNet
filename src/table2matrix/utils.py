# -*- coding: utf-8 -*-
import re
import numpy as np
from itertools import permutations
import networkx as nx
from dateutil.parser import parse as date_parse

from collections import defaultdict


####################################
# Utils for encode original features
####################################
def positional_encode(row_idx, col_idx, row_count, col_count):
    feats = []

    feats.extend([row_idx/row_count, col_idx/col_count])
    feats.extend([(row_count-row_idx)/row_count,
                  (col_count-col_idx)/col_count])
    feats.extend([np.exp(-row_idx), np.exp(-col_idx)])

    return feats


class Encoder:
    def __init__(self):
        self.format = formatParse()

    def encode(self, cellAttr: dict):
        not_text = None
        vec = []
        for k, v in cellAttr.items():
            if not hasattr(self, k):
                continue
            elif k == "NumberFormatString":
                sub_vec, not_text = getattr(self, k)(v, cellAttr["Text"])
            else:
                sub_vec = getattr(self, k)(k, v)
            vec.extend(sub_vec)
        return vec, not_text

    # Text statistic
    def Text(self, k, v):
        sub1 = int(v == "")
        sub2 = len(v)
        sub3 = len(re.sub("\D", "", v)) / len(v) if len(v) > 0 else 0
        sub4 = int("%" in v)
        sub5 = int("." in v)
        sub6 = len(re.sub("[a-zA-Z]", "", v)) / len(v) if len(v) > 0 else 0
        sub_vec = [sub1, sub2, sub3, sub4, sub5, sub6]
        return sub_vec

    # Format
    def NumberFormatString(self, f, t):
        sub_vec, not_text = self.format.vec(f, t)
        return sub_vec, not_text

    # Cell Format
    def FillColor(self, k, v):
        sub_vec = self._RgbaToVec(v)
        return sub_vec

    def TopBorderStyleType(self, k, v):
        sub_vec = [0] if v.strip().lower() == "none" else [1]
        return sub_vec

    def LeftBorderStyleType(self, k, v):
        sub_vec = [0] if v.strip().lower() == "none" else [1]
        return sub_vec

    def RightBorderStyleType(self, k, v):
        sub_vec = [0] if v.strip().lower() == "none" else [1]
        return sub_vec

    def BottomBorderStyleType(self, k, v):
        sub_vec = [0] if v.strip().lower() == "none" else [1]
        return sub_vec

    def TopBorderColor(self, k, v):
        sub_vec = self._RgbaToVec(v)
        return sub_vec

    def BottomBorderColor(self, k, v):
        sub_vec = self._RgbaToVec(v)
        return sub_vec

    def LeftBorderColor(self, k, v):
        sub_vec = self._RgbaToVec(v)
        return sub_vec

    def RightBorderColor(self, k, v):
        sub_vec = self._RgbaToVec(v)
        return sub_vec

    # Font format
    def FontColor(self, k, v):
        sub_vec = self._RgbaToVec(v)
        return sub_vec

    def FontBold(self, k, v):
        sub_vec = [int(v)]
        return sub_vec

    def FontUnderlineType(self, k, v):
        sub_vec = [0] if v.strip().lower() == "none" else [1]
        return sub_vec

    def FontSize(selfself, k, v):
        sub_vec = [float(v)]
        return sub_vec

    # Others
    def HasFormula(self, k, v):
        sub_vec = [int(v)]
        return sub_vec

    def IsNumber(self, k, v):
        sub_vec = [int(v)]
        return sub_vec

    def IndentLevel(self, k, v):
        sub_vec = [float(v)]
        return sub_vec

    def _RgbaToVec(self, color):
        r = int("0x" + color[1:3], 16) / 255
        g = int("0x" + color[3:5], 16) / 255
        b = int("0x" + color[5:7], 16) / 255
        a = int("0x" + color[7:], 16) / 255

        return [r, g, b, a]


class formatParse:
    def __init__(self):
        self.NumberFormat = ["0.00;[Red]0.00", "0.00_);(0.00)",
                             "0.00_);[Red](0.00)", "# ???/???", "# ?/2",
                             "# ?/4", "# ?/8", "# ??/16", "# ?/10", "# ??/100",
                             "0", "0.00", "#,##0", "#,##0.00", "0.00E+00", "# ?/?",
                             "# ??/??", "#,##0 ;(#,##0)", "#,##0 ;[Red](#,##0)",
                             "#,##0.00;(#,##0.00)", "#,##0.00;[Red](#,##0.00)",
                             "##0.0E+0"]

        self.TextFormat = ["00000", "00000-0000", "@",
                           "[<=9999999]###-####;(###) ###-####", "000-00-0000"]

        self.DateTimeFormat = ["[$-F800]dddd, mmmm dd, yyyy", "yyyy-mm-dd;@",
                               "m/d;@", "m/d/yy;@", "mm/dd/yy;@", "[$-409]d-mmm;@", "[$-409]d-mmm-yy;@",
                               "[$-409]dd-mmm-yy;@", "[$-409]mmm-yy;@", "[$-409]mmmm-yy;@", "[$-409]mmmm d, yyyy;@",
                               "[$-409]m/d/yy h:mm AM/PM;@", "m/d/yy h:mm;@", "[$-409]mmmmm;@", "[$-409]mmmmm-yy;@",
                               "m/d/yyyy;@", "[$-409]d-mmm-yyyy;@", "[$-F400]h:mm:ss AM/PM", "h:mm;@",
                               "[$-409]h:mm AM/PM;@",
                               "h:mm:ss;@", "[$-409]h:mm:ss AM/PM;@", "mm:ss.0;@",
                               "[h]:mm:ss;@", "[$-409]m/d/yy h:mm AM/PM;@", "m/d/yy h:mm;@", "d/m/yyyy",
                               "d-mmm-yy", "d-mmm", "mmm-yy", "h:mm tt", "h:mm:ss tt", "H:mm", "H:mm:ss",
                               "m/d/yyyy H:mm", "mm:ss", "[h]:mm:ss", "mmss.0"]

        self.PercentageFormat = ["0%", "0.00%"]

        self.CurrencyFormat = ["$#,##0.00", "$#,##0.00;[Red]$#,##0.00",
                               "$#,##0.00_);($#,##0.00)", "$#,##0.00_);[Red]($#,##0.00)",
                               "$#,##0_);[Red]($#,##0)", "_($* #,##0.00_);_($* (#,##0.00);_($* \"-\"??_);_(@_)",
                               "$#,##0_);($#,##0)"]

        self.format_type = {"Number": 0, "Text": 1, "Other": 2,
                            "DateTime": 3, "Percentage": 4, "Currency": 5}

    def vec(self, text_format: str, text: str):
        sub_vec = np.zeros(6)
        if text_format in self.NumberFormat:
            if re.match("\d", text):
                sub_vec[self.format_type["Number"]] = 1
        elif text_format in self.TextFormat:
            sub_vec[self.format_type["Text"]] = 1
        elif text_format in self.DateTimeFormat:
            sub_vec[self.format_type["DateTime"]] = 1
        elif text_format in self.PercentageFormat:
            sub_vec[self.format_type["Percentage"]] = 1
        elif text_format in self.CurrencyFormat:
            sub_vec[self.format_type["Currency"]] = 1
        if sum(sub_vec) == 0:
            sub_vec = self.textParse(text)

        assert sum(sub_vec) == 1
        if sub_vec[self.format_type["Text"]] != 1:
            not_text = True
        else:
            not_text = False
        return sub_vec, not_text

    def textParse(self, text: str):
        sub_vec = np.zeros(6)
        text = text.strip().lower()
        if len(text) == 0:
            sub_vec[self.format_type["Other"]] = 1
        elif self.isnumber(text):
            sub_vec[self.format_type["Number"]] = 1
        elif (len(text) > 1) & (self.re_search("%", text)):
            if self.isnumber(text[1:-1]):
                sub_vec[self.format_type["Percentage"]] = 1
            else:
                sub_vec[self.format_type["Other"]] = 1
        elif (len(text) > 1) & (self.re_search("[$,¢,£,€,¥]", text)):
            if self.isnumber(text[1:-1]):
                sub_vec[self.format_type["Currency"]] = 1
            else:
                sub_vec[self.format_type["Other"]] = 1
        elif self.is_date(text):
            sub_vec[self.format_type["DateTime"]] = 1
        elif re.match("[a-z]", text):
            sub_vec[self.format_type["Text"]] = 1
        else:
            sub_vec[self.format_type["Other"]] = 1

        return sub_vec

    def re_search(self, pattern, text):
        res = re.search(pattern, text)

        if res is None:
            return False
        else:
            if res[0] == " ":
                return False
            else:
                return True

    def is_date(self, text):
        try:
            date_parse(text, fuzzy=False)
            return True

        except:
            return False

    def isnumber(self, s):
        s = s.replace(",", "")
        try:
            float(s)
            return True
        except:
            try:
                int(s)
                return True
            except:
                return False


def merged_cells(merged_regions, M, N):
    """
    :param merged_regions: list, merged regions from table file
    :param N: int, the number of row
    :param M: int, the number of col
    :return: list of list, contains all merged group's nodes respectively; list, each group's leader.
    """
    merged_groups, merged_groups_leaders = [], []
    oor = False  # out of range -> oor
    for corners in merged_regions:
        tmp_group = []
        fr = max(0, corners["FirstRow"])
        lr = min(N-1, corners["LastRow"])
        fc = max(0, corners["FirstColumn"])
        lc = min(M-1, corners["LastColumn"])

        for r in range(fr, lr + 1):
            for c in range(fc, lc + 1):
                node_idx = r * M + c
                tmp_group.append(node_idx)

        merged_groups.append(tmp_group)
        merged_groups_leaders.append(min([node for node in tmp_group]))
    return merged_groups, merged_groups_leaders


def heur_edges(non_textual_matrix, relations):
    alpha = 0.5
    max_edge = 3
    min_edge = 0.5
    N, M = non_textual_matrix.shape
    Nodes = np.arange(N*M).reshape(N, M)

    def R_edge(i, X): return X*(min_edge + (max_edge-min_edge)*np.exp(-i))
    def P_wgt(X): return np.array([1/x for x in range(1, X+1)])
    def Edge_sample(edges, size, p): return np.random.choice(
        edges, size=size, replace=True, p=p)

    G = nx.DiGraph()

    # Add relation edges
    for edge in get_relation_edges(M, relations):
        G.add_edge(*edge)

    # Add Row Heur
    row_p_wgt = P_wgt(M)
    for r, r_nodes in enumerate(Nodes):
        add_num_edge = int(R_edge(r, M))
        info_wgt = alpha*(row_p_wgt) + (1-alpha)*non_textual_matrix[r, :]
        info_wgt /= np.sum(info_wgt)

        edge_nodes = np.random.choice(r_nodes, size=2*add_num_edge, p=info_wgt)
        edges = edge_nodes.reshape(add_num_edge, 2)
        G.add_edges_from(edges)

    # Add Col Heur
    col_p_wgt = P_wgt(N)
    for c, c_nodes in enumerate(Nodes.T):
        add_num_edge = int(R_edge(c, N))
        info_wgt = alpha*(col_p_wgt) + (1-alpha)*non_textual_matrix[:, c]
        info_wgt /= np.sum(info_wgt)

        edge_nodes = np.random.choice(c_nodes, size=2*add_num_edge, p=info_wgt)
        edges = edge_nodes.reshape(add_num_edge, 2)
        G.add_edges_from(edges)

    return np.array(list(G.edges)).T


def get_relation_edges(M, relations):
    edges = []
    if len(relations) == 0:
        return edges

    for r in relations:
        v = list(r.values())
        s = v[0]*M + v[1]
        t = v[2]*M + v[3]

        if s < 0 or t < 0:
            continue
        edges.append((s, t))
        if not v[5]:
            edges.append((t, s))

    return edges


def tlbr_edges(N, M):
    g = nx.generators.lattice.grid_2d_graph(N, M)
    def to_node(i, j): return i*M + j

    edge_index = [[to_node(*ij) for ij in edge] for edge in list(g.edges())]
    edge_index = np.array(edge_index).T

    return edge_index
