import torch
from torch import nn
from torch.nn import functional as F

from .utils import mlp, LSTMCell, LSTMCell_PB

import itertools


class Language(nn.Module):
    """
    creating language module with LSTM or LSTM with parametric bias
    """
    def __init__(self, in_sizes, is_lang, is_pb, dim, layers, dropout_rate=0.5):
        super(Language, self).__init__()
        self.is_pb = is_pb
        self.num_layers = len(in_sizes)
        self.lang_net = nn.ModuleList()
        for l in range(len(layers)):
            if self.is_pb:
                layer = LSTMCell_PB(in_sizes[l], layers[l]['hid_size'], layers[l]['pb_size'], dropout_rate=dropout_rate)
            else:
                layer = LSTMCell(in_sizes[l], layers[l]['hid_size'], dropout_rate=dropout_rate)
            self.lang_net.append(layer)
    def forward(self, l, sxs, pb):
        ss = []
        gates = []
        for layer_idx in range(self.num_layers):
            if self.is_pb:
                h, c, gate = self.lang_net[layer_idx](l, sxs, pb)
            else:
                h, c, gate = self.lang_net[layer_idx](l, sxs)
            ss.append([h, c])
            gates.append(gate)
        return ss, gates



