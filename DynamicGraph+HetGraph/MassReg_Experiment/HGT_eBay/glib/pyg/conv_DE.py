from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class DysatConv(MessagePassing):

    def __init__(self, conv_struct, conv_ts,
                 fc, edge_index_ts,
                 dropout):
        super().__init__()

        self.conv_struct = conv_struct
        self.conv_ts = conv_ts

        if fc is not None:
            self.fc = fc
            self.norm = nn.LayerNorm(normalized_shape=fc.out_features)
            self.drop = nn.Dropout(p=dropout)
        else:
            self.fc = None

        self.edge_index_ts = edge_index_ts

    def forward(self, x, edge_index):
        x = self.conv_struct(x, edge_index)

        if self.fc is not None:
            x = self.fc(x)
            x = self.norm(x)
            x = torch.relu(x)
            x = self.drop(x)

        x = self.conv_ts(x, self.edge_index_ts)
        return x


class DysatGCN(DysatConv):

    def __init__(self, in_hid, out_hid, edge_index_ts, dropout):

        conv_struct = GCNConv(in_hid, out_hid)
        fc = nn.Linear(out_hid, out_hid)
        conv_ts = GCNConv(out_hid, out_hid)

        super(DysatGCN, self).__init__(
            conv_struct, conv_ts, fc, edge_index_ts, dropout=dropout)


class DysatGCN_v0(DysatConv):

    def __init__(self, in_hid, out_hid, edge_index_ts, dropout):

        conv_struct = GCNConv(in_hid, out_hid)
        conv_ts = GCNConv(out_hid, out_hid)

        super(DysatGCN_v0, self).__init__(
            conv_struct, conv_ts, None, edge_index_ts, dropout=dropout)


class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout, 
                 num_node_type, num_edge_type, edge_index_ts):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(
                in_hid, out_hid // n_heads, heads=n_heads, dropout=dropout)
        elif self.conv_name == 'dysat-gcn':
            assert edge_index_ts is not None
            self.base_conv = DysatGCN(
                in_hid, out_hid, edge_index_ts=edge_index_ts, dropout=dropout)
        elif self.conv_name == 'dysat-gcn-v0':
            assert edge_index_ts is not None
            self.base_conv = DysatGCN_v0(
                in_hid, out_hid, edge_index_ts=edge_index_ts, dropout=dropout)
        else:
            raise NotImplementedError('unknown conv name %s' % conv_name)

    def forward(self, x, edge_index, node_type, edge_type, *args):
        if self.conv_name in {'gcn', 'gat'} or \
                self.conv_name.startswith('dysat-'):
            return self.base_conv(x, edge_index)
        elif self.conv_name in {'het-emb'}:
            return self.base_conv(x, edge_index, node_type, edge_type)
        raise NotImplementedError('unknown conv name %s' % self.conv_name)

