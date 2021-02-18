import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class BIGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers, dropout):
        super(BIGRU, self).__init__()

        self.gru = nn.GRU(in_dim, hid_dim, num_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=True)
        self.out_layer = nn.Linear(2*hid_dim, in_dim)

    def forward(self, padded_x, mask):
        """padded_x has size=[batch, max_seq_len, feat_dim],
            mask has size=[batchm ,max_seq, max_seq]"""
        lengths = torch.sum(mask[:, :, 0], dim=1).int()

        packedSeq = pack_padded_sequence(padded_x, lengths,
                                         batch_first=True, enforce_sorted=False)

        output, _ = self.gru(packedSeq)
        # output.data: [batch, max_seq_len, 2*hid_dim]
        unpacked = pad_packed_sequence(output, batch_first=True)[0]

        return F.relu(self.out_layer(unpacked))
