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
import math

class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.emb(t))


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
        elif self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types=num_node_type, num_relations=num_edge_type, n_heads=n_heads, dropout=dropout, use_norm=True, use_RTE=False)
        else:
            raise NotImplementedError('unknown conv name %s' % conv_name)

    def forward(self, x, edge_index, node_type, edge_type, *args):
        if self.conv_name in {'gcn', 'gat'} or \
                self.conv_name.startswith('dysat-'):
            return self.base_conv(x, edge_index)
        elif self.conv_name in {'het-emb'}:
            return self.base_conv(x, edge_index, node_type, edge_type)
        elif self.conv_name == 'hgt':
            return self.base_conv(x, node_type, edge_index, edge_type, edge_time=None)
        raise NotImplementedError('unknown conv name %s' % self.conv_name)

