import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.BaseModel import BaseModel
from src.encoder.gine import GINEEncoder
from src.encoder.bigru import BIGRU
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import batched_negative_sampling


class gine_pool_bigru(BaseModel):
    def __init__(self, feat_dim, hid_dim, num_class, dropout, device, args):
        super(gine_pool_bigru, self).__init__(
            feat_dim, hid_dim, num_class, dropout, device, args)

        self.row_encoder = BIGRU(hid_dim, hid_dim, args.gru_layers, dropout)
        self.col_encoder = BIGRU(hid_dim, hid_dim, args.gru_layers, dropout)
        self.args = args
        if args.graph_type == 'original' and args.gcn_pos == 'inner':
            self.gcn = GINEEncoder(feat_dim, hid_dim, dropout)
            class_dim = hid_dim*5
        elif args.graph_type == 'original' and args.gcn_pos == 'outer':
            self.gcn = GINEEncoder(feat_dim, hid_dim, dropout)
            self.outer_gcn = SAGEConv(hid_dim*5, hid_dim)
            class_dim = hid_dim
        elif args.graph_type == 'wnsim' and args.gcn_pos == 'inner':
            self.gcn = SAGEConv(feat_dim, hid_dim)
            class_dim = hid_dim*5
        elif args.graph_type == 'wnsim' and args.gcn_pos == 'outer':
            self.outer_gcn = SAGEConv(hid_dim*4, hid_dim)
            class_dim = hid_dim

        self.classifier = self.set_classifier(class_dim)
        self.dp = nn.Dropout(dropout)
        self.sage = SAGEConv(hid_dim*5, hid_dim)

    def forward(self, batch):
        row_x, row_mask, col_x, col_mask, row_lens, col_lens = self.base_forward(
            batch)

        # 1. <<<self-attention encoding>>>
        row_emb = self.row_encoder(row_x, row_mask)
        col_emb = self.col_encoder(col_x, col_mask)
        gru_emb = self.RowCol_emb_flatten(
            row_emb, col_emb, row_lens, col_lens, batch)  # ==> [Nodes, 2*hid_dim]
        gru_emb = self.dp(gru_emb)

        original_emb = self.original_feats(batch)  # ==> [Nodes, hid_dim]
        original_emb = self.dp(original_emb)

        row_col_emb = self.row_col_pooling(batch)  # ==> [Nodes, hid_dim]
        row_col_emb = self.dp(row_col_emb)

        if self.args.graph_type == 'original' and self.args.gcn_pos == 'inner':
            gnn_emb = self.gcn(self.pyg_batch(batch))  # ==> [Nodes, hid_dim]
            gnn_emb = self.dp(gnn_emb)
            cat_embs = torch.cat(
                (original_emb, gru_emb, row_col_emb, gnn_emb), dim=1)
        elif self.args.graph_type == 'original' and self.args.gcn_pos == 'outer':
            gnn_emb = self.gcn(self.pyg_batch(batch))  # ==> [Nodes, hid_dim]
            gnn_emb = self.dp(gnn_emb)
            cat_embs = torch.cat(
                (original_emb, gru_emb, row_col_emb, gnn_emb), dim=1)
            edge_index = batch.wn_batch(self.device).edge_index
            cat_embs = self.outer_gcn(cat_embs, edge_index)
        elif self.args.graph_type == 'wnsim' and self.args.gcn_pos == 'inner':
            gnn_emb = self.gcn(batch.pyg_batch(self.device).x,
                               batch.pyg_batch(self.device).edge_index)
            gnn_emb = self.dp(gnn_emb)
            cat_embs = torch.cat(
                (original_emb, gru_emb, row_col_emb, gnn_emb), dim=1)
        elif self.args.graph_type == 'wnsim' and self.args.gcn_pos == 'outer':
            cat_embs = torch.cat((original_emb, gru_emb, row_col_emb), dim=1)
            edge_index = batch.pyg_batch(self.device).edge_index
            cat_embs = self.outer_gcn(cat_embs, edge_index)
            cat_embs = self.dp(cat_embs)

        return cat_embs, batch

    def wn_gcn(self, embeds, batch):
        edge_index = batch.wn_batch(self.device).edge_index
        out_embs = self.sage(embeds, edge_index)
        return out_embs, batch
