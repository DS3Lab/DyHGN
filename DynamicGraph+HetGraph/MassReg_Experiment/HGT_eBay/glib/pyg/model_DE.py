import torch
import torch.nn as nn
from glib.pyg.conv_DE import GeneralConv
import tqdm

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.l1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
    def forward(self, x):
        r_out, (h_n, h_c) = self.l1(x, None) #None represents zero initial hidden state
        return r_out[:, -1, :]

class GNN(nn.Module):

    def __init__(self, n_in, n_hid, n_layers, n_heads, dropout, 
                 conv_name, num_node_type, num_edge_type,
                 edge_index_ts=None):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.conv_name = conv_name

        self.gcs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        for i in range(n_layers):
            layer = GeneralConv(
                conv_name, 
                in_hid=n_in if i==0 else n_hid, 
                out_hid=n_hid, n_heads=n_heads, dropout=dropout,
                num_node_type=num_node_type, num_edge_type=num_edge_type,
                edge_index_ts=edge_index_ts
            )
            self.gcs.append(layer)
            layer = nn.LayerNorm(n_hid)
            self.norms.append(layer)
    
    def forward(self, x, edge_index, node_type=None, edge_type=None):
        if not isinstance(edge_index, (list, tuple)):
            edge_index = [edge_index]
        if not isinstance(edge_type, (list, tuple)):
            edge_type = [edge_type]

        if edge_type is None:
            edge_type = [None] * len(edge_index)

        assert len(edge_index) == len(edge_type)

        if len(self.gcs) > len(edge_index):
            if len(edge_index) == 1:
                edge_index = [edge_index[0]] * len(self.gcs)
                edge_type = [edge_type[0]] * len(self.gcs)
            else:
                raise RuntimeError(
                    'Mismatch layer number gcs %d and edge_index %d!' % (
                        len(self.gcs), len(edge_index)))
        for conv, norm, ei, et in zip(self.gcs, self.norms, edge_index, edge_type):
            x = conv(x, ei, node_type, et)
            x = self.dropout(x)
            x = norm(x)
            x = torch.relu(x)
        return x


class HetNet(nn.Module):
    def __init__(self, gnn, num_feature, num_embed, node_count, emb_dim,
                 n_hidden=128, num_output=2, dropout=0.5, n_layer=1):
        super(HetNet, self).__init__()
        if gnn is None:
            self.gnn = None
            self.fc1 = nn.Linear(num_feature, n_hidden)    
        else:
            self.gnn = gnn
            self.fc1 = nn.Linear(num_feature + num_embed, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.lyr_norm1 = nn.LayerNorm(normalized_shape=n_hidden)

        if n_layer == 1:
            pass
        elif n_layer == 2:
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.lyr_norm2 = nn.LayerNorm(normalized_shape=n_hidden)
        else:
            raise NotImplementedError()
        self.n_layer = n_layer
        
        self.lstmmodel = LSTMModel(input_size=emb_dim, hidden_size=emb_dim)
        
        self.out   = nn.Linear(n_hidden, num_output)
        
        ## adding embedding TODO: implement emb_dim and se_prop as parameters
        self.se_prop=0.36
        self.emb_dim=emb_dim
        
        self.s_emb_dim = int(self.se_prop*self.emb_dim)
        self.t_emb_dim = self.emb_dim - int(self.se_prop*self.emb_dim)
        self.emb_dropout = 0.4
        
        self.numNode = node_count #number of distinct nodes
        self.numEdge = 4 #different types of edges
        
        self.ent_embs  = nn.Embedding(self.numNode, self.s_emb_dim) 
        self.rel_embs = nn.Embedding(self.numEdge, self.s_emb_dim + self.t_emb_dim)
        
        # Creating and initializing the temporal embeddings for the entities 
        self.create_time_embedds()
        
        # Setting the non-linearity to be used for temporal part of the embedding
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        
    
    def create_time_embedds(self):
        
        # frequency embeddings for the entities
        self.w_freq = nn.Embedding(self.numNode, self.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.numNode, self.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.w_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)

        # phi embeddings for the entities
        self.w_phi = nn.Embedding(self.numNode, self.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.numNode, self.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.w_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)

        # amplitude embeddings for the entities
        self.w_amp = nn.Embedding(self.numNode, self.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.numNode, self.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.w_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        
            
    def get_time_embedd(self, entities, week, day):

        w = self.w_amp(entities)*self.time_nl(self.w_freq(entities)*week + self.w_phi(entities))
        d = self.d_amp(entities)*self.time_nl(self.d_freq(entities)*day + self.d_phi(entities))

        return w+d

    def getEmbeddings(self, heads, rels, tails, weeks, days, intervals = None):
        weeks = weeks.view(-1,1)
        days = days.view(-1,1)
        
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)        
        h_t = self.get_time_embedd(heads, weeks, days)
        t_t = self.get_time_embedd(tails, weeks, days)
        
        h = torch.cat((h,h_t), 1)
        t = torch.cat((t,t_t), 1)
        return h,r,t
        
    def forward(self, x, edge, heads, rels, tails, weeks, days, edge_index=None, **kwargs):
        if edge_index is None:
            mask, x, edge_index, *args = x
        else:
            args = tuple()
            mask = torch.arange(x.shape[0])
        
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, weeks, days)
        
        scores = (h_embs * r_embs) * t_embs
        
        emb = torch.zeros((x.shape[0], self.emb_dim)).cuda()
        
        for src, edge_grp in tqdm.tqdm(edge.groupby('src'), position=0, mininterval=10):
          emb[src,:] = self.lstmmodel(scores[edge_grp.index].view(-1, scores[edge_grp.index].shape[0], self.emb_dim))
        
        for dst, edge_grp in tqdm.tqdm(edge.groupby('dst'), position=0, mininterval=10):
          emb[dst,:] = self.lstmmodel(scores[edge_grp.index].view(-1, scores[edge_grp.index].shape[0], self.emb_dim))
        
        #self.emb=emb
        x = torch.cat((x,emb), 1)
            
        if self.gnn is not None:
            x0 = x
            x = self.gnn(x0, edge_index, *args, **kwargs)
            x = torch.tanh(x)
            x = torch.cat([x0, x], 1)

        x = x[mask]
        x = self.fc1(x)
        x = self.lyr_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        if self.n_layer == 1:
            pass
        elif self.n_layer == 2:
            x = self.fc2(x)
            x = self.lyr_norm2(x)
            x = torch.relu(x)
            x = self.dropout(x)
        else:
            raise NotImplementedError()

        x = self.out(x)
        return x

    def saveEmbedding2(self, epoch, name):
        torch.save(self.emb, name+"-emb"+str(epoch)+".pt")