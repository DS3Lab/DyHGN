import os
import glob
import logging
from itertools import product

from numpy.core.numeric import False_

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger('exp')
logger.setLevel(logging.INFO)

DIR_DATA = os.path.abspath(os.path.dirname(__file__)) + \
           '/../../../data/shared/ebay-xfraud/'
CUDA_LAUNCH_BLOCKING = 1

import fire
import tqdm
import numpy as np
from scipy.special import softmax
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, average_precision_score, accuracy_score,
    precision_score, accuracy_score, classification_report, precision_recall_curve
)
from sklearn.model_selection import train_test_split

# torch
import torch
import torch.nn as nn

from ignite.utils import convert_tensor
# ebay exp
from glib.pyg.model_DE_HGT import HetNet as Net, GNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_device = torch.cuda.device_count()

def create_enc_dec_node(edge):
    enc = dict((e, i) for i, e in enumerate(edge['src'].unique()))
    for e in edge['dst'].unique():
        enc[e] = len(enc)
    dec = dict((v, k) for k, v in enc.items())
    return enc, dec


def get_data(debug=False, task="txn",
             ts_snapshot=False, get_node_type=False):

    path_x = f'{DIR_DATA}/{task}/feat.parquet'
    path_y = f'{DIR_DATA}/{task}/label.parquet'
    path_node = f'{DIR_DATA}/{task}/node.parquet'
    path_edge = f'{DIR_DATA}/{task}/edge.parquet'
    x = pd.read_parquet(path_x)
    y = pd.read_parquet(path_y)
    node = pd.read_parquet(path_node)
    edge = pd.read_parquet(path_edge)
    
    edge = edge.reset_index(drop=True)

    if debug:
        edge = edge.sample(n=10000)

    if ts_snapshot:
        """
        edge_new = pd.merge(
            edge, x[['node_id', 'ts_wk']],
            left_on='src', right_on='node_id')
        """
        edge_new = edge.copy()
        edge_new['dst_origin'] = edge_new['dst']
        edge_new['dst'] = edge_new['dst_origin'].astype(np.int64) * 100 + edge_new['ts_wk']

        edge_to_encode = pd.concat([
            edge[['src', 'dst']],
            edge_new[['src', 'dst']]
            ])

        enc, dec = create_enc_dec_node(edge_to_encode)
        x['node_id'] = x['node_id'].map(enc)
        y['node_id'] = y['node_id'].map(enc)

        edge = edge_new
        edge['src'] = edge['src'].map(enc)
        edge['dst'] = edge['dst'].map(enc)

        g = edge[['src', 'dst']].values
        g = np.concatenate([g, edge[['dst', 'src']].values])

        edge_ts = []
        for dst_origin, edge_grp in tqdm.tqdm(
                edge_new.groupby('dst_origin'),
                total=edge_new['dst_origin'].nunique(), desc='ts-edge-init'):
            for d0 in edge_grp['dst'].unique():
                edge_ts.append([d0, enc[dst_origin]])
                edge_ts.append([enc[dst_origin], d0])
        g_ts = np.asarray(edge_ts)
    else:
        enc, dec = create_enc_dec_node(edge)

        x['node_id'] = x['node_id'].map(enc)
        y['node_id'] = y['node_id'].map(enc)
        edge['src'] = edge['src'].map(enc)
        edge['dst'] = edge['dst'].map(enc)

        g = edge[['src', 'dst']].values
        g = np.concatenate([g, edge[['dst', 'src']].values])
        g_ts = None

    df_tpl = pd.DataFrame(dict(node_id=np.arange(len(enc))))
    
    x = pd.merge(df_tpl, x, how='left', on='node_id')
    x.fillna(0., inplace=True)

    y = y.fillna(0)
    y = pd.merge(df_tpl, y, how='left', on='node_id')
    y = y.fillna(-1)
    y["risky"] = y["risky"].astype(int)

    num_node_type = len(set(node["node_type"]))
    node_type = np.zeros(len(x))
    if get_node_type:
        print("getting node type....")
        for i in range(len(x)):
          try:
            if len(node.loc[node["node_id"] == dec[i]])==0:
              if len(node["node_type"].loc[node["node_id"] == int(str(dec[i])[:-2])])==0:
                node_type[i] = 0
              else:
                node_type[i] = node["node_type"].loc[node["node_id"] == int(str(dec[i])[:-2])]
            else:
              node_type[i] = node["node_type"].loc[node["node_id"] == dec[i]]
          except:
            node_type[i] = 0
            continue
    node_type = convert_tensor(torch.LongTensor(node_type), device=device)
    return g, x, y, g_ts, edge, node_type


def get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main(path_result='result-mass-reg.csv',
         conv_name='dysat-gcn',
         n_hid=256, n_heads=8, n_layers=4, dropout=0.1,
         optimizer='adamw',
         max_epochs=8,
         valid_ratio=0.2, patience=64, emb_dim=30,
         seed=2020, verbose=False, task="txn", debug=False):

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    assert task in ["txn","account"], "unautorized task"
    
    assert os.path.exists(f'{DIR_DATA}/{task}/node.parquet'), \
    f'Please untar the file in {DIR_DATA}!'
    
    results_file = 'results_'+task+'_lstm_hgt'+str(seed)+'.txt'
    # Remove previous results
    for f in glob.glob(results_file):
        os.remove(f)

    logger.info('Seed %d, Device %s', seed, device)

    dysat = conv_name.startswith('dysat')

    stats = dict(
        n_hid=n_hid, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        conv_name=conv_name, optimizer=str(optimizer),
        max_epochs=max_epochs, patience=patience,
        seed=seed, task=task
    )
    logger.info('Param %s', stats)
    
    g, x, y, g_ts, edge, node_type = get_data(ts_snapshot=dysat, task=task, debug=debug, get_node_type=True)

    print(x.shape, y.shape, edge.shape)
    node_count = len(np.unique(g[:, 0]))+len(np.unique(g[:, 1]))
    edge_count = g.shape[0]
    logger.info('#node %d, #edge %d', node_count, edge_count)
    if g_ts is not None:
        logger.info('\t#edge-ts %d', g_ts.shape[0])

    num_feat = len([c for c in x if c.startswith('feat')])

    if not conv_name:
        gnn = None
        edge_index_ts = None
    else:
        edge_index_ts = (
            None if g_ts is None
            else convert_tensor(
                torch.LongTensor(g_ts.T), device=device
            )
        )
        gnn = GNN(n_in=num_feat+2*emb_dim,
                  n_hid=n_hid, n_layers=n_layers,
                  n_heads=n_heads, dropout=dropout,
                  conv_name=conv_name,
                  num_node_type=1, num_edge_type=1,
                  edge_index_ts=edge_index_ts)

    model = Net(
        gnn, num_feature=num_feat+2*emb_dim, n_heads=n_heads,
        num_embed=0 if gnn is None else n_hid, node_count=node_count, emb_dim=emb_dim,
        n_hidden=n_hid, num_output=1, dropout=dropout, n_layer=2
    )
    model = model.to(device)
        
    
    # optimizer = torch.optim.AdamW(model.parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=1e-3, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=128,
        num_training_steps=max_epochs)

    logger.info('Risky\n%s', y.risky.value_counts())

    y0 = convert_tensor(
        torch.FloatTensor(y['risky'].values),
        device=device
    )

    mask_train_origin = mask_train = (y0>=0) & convert_tensor(
        torch.BoolTensor((x['phase'] == 'TRAIN')), device=device)

    if task == "account_meth2":
        mask_train_idx = x.index[mask_train.cpu()].tolist()
        mask_train_idx, mask_valid_idx = train_test_split(
            mask_train_idx, test_size=valid_ratio, random_state=seed)

        mask_train = convert_tensor(
            torch.BoolTensor(x.index.isin(mask_train_idx)), device=device
        )
        mask_valid = convert_tensor(
            torch.BoolTensor(x.index.isin(mask_valid_idx)), device=device
        )
    else: 
        if task=="txn":
            wk_thresh = 3
        elif task=="account":
            wk_thresh = 2
        
        mask_train = (y0>=0) & convert_tensor(
            torch.BoolTensor((x['phase'] == 'TRAIN')), device=device) & convert_tensor(
                torch.BoolTensor((x['ts_wk'] <= wk_thresh)) , device=device)
        
        mask_valid = (y0>=0) & convert_tensor(
            torch.BoolTensor((x['phase'] == 'TRAIN')), device=device) & convert_tensor(
                torch.BoolTensor((x['ts_wk'] > wk_thresh)) , device=device)

    mask_test = (y0>=0) & convert_tensor(
        torch.BoolTensor((x['phase'] == 'TEST')), device=device)
    
    
    x = convert_tensor(
        torch.FloatTensor(x[[c for c in x if c.startswith('feat')]].values),
        device=device
    )

    y0_train = y0[mask_train]

    edge_index = convert_tensor(
        torch.LongTensor(g.T), device=device
    )
    
    heads = torch.tensor(edge["src"].values).long().to(device)
    rels = torch.tensor(edge["edge_type"].values).long().to(device) 
    tails = torch.tensor(edge["dst"].values).long().to(device)
    weeks = torch.tensor(edge["ts_wk"].values).float().to(device)
    
    edge_type = torch.cat((torch.tensor(edge["edge_type"].values), torch.tensor(edge["edge_type"].values))).to(device) #to match edge_index
    node_type = node_type.to(device)
    
    def train(loss0_fn=nn.BCEWithLogitsLoss()):
        model.train()
        optimizer.zero_grad()

        out = model((mask_train, x, edge_index, edge_index_ts), edge, heads, rels, tails, weeks, node_type, edge_type)
        y0_yhat = out[:,0]
        loss = loss0_fn(y0_yhat, y0_train)

        loss.backward()
        optimizer.step()

    # def metric_regression(y0_yhat, y0):
    #     y0_yhat, y0 = [np.asarray(e.cpu()) for e in [y0_yhat, y0]]
    #     return mean_squared_error(y0, y0_yhat)

    # def metric_multilabel(yhat, y):
    #     yhat, y = [np.asarray(e.cpu()) for e in [yhat, y]]
    #     rval = {}
    #     for i in range(y.shape[1]):
    #         rval[f'ap_{i}'] = average_precision_score(y[:, i], yhat[:, i])
    #         rval[f'roc_{i}'] = roc_auc_score(y[:, i], yhat[:, i])
    #         rval[f'accuracy_{i}'] = accuracy_score(y[:, i], yhat[:, i]>0.5)
    #     return rval

    def metric_multiclass(yhat, y):
        yhat, y = [np.asarray(e.cpu()) for e in [yhat, y]]
        yhat = softmax(yhat, axis=1)
        auc = roc_auc_score(y, yhat, multi_class='ovr')
        yhat = yhat.argmax(axis=1)
        if verbose:
            print(classification_report(y, yhat))
        return {'accuracy': accuracy_score(y, yhat), 'auc': auc}

    def metric_binary_class(yhat, y):
        yhat, y = [np.asarray(e.cpu()) for e in [yhat, y]]
        if verbose:
            print(classification_report(y, yhat>0.5))
        return {'auc': roc_auc_score(y, yhat),
                'ap': average_precision_score(y, yhat)}

    @torch.no_grad()
    def test(epoch, mask):
        model.eval()
        out = model((mask, x, edge_index, edge_index_ts), edge, heads, rels, tails, weeks, node_type, edge_type)
        y0_yhat = out[:,0]
        y0_test = y0[mask]
        m0 = metric_binary_class(y0_yhat, y0_test.long())
        if verbose:
            logger.info(f'Epoch {epoch}: {m0}')
        return m0

    best = (0, -1)
    pbar = tqdm.tqdm(total=max_epochs, desc='train-stg0')
    for epoch in range(1, max_epochs+1):
        pbar.update(1)

        train()
        m0 = test(epoch, mask_valid)
        for k, v in m0.items():
            stats[k] = v
        metric = stats['ap']
        
        pbar.set_description(f'train-stg0 epoch {epoch} - auc {m0["auc"]:.4f} ap {m0["ap"]:.4f}')
        
        with open(results_file, 'a') as f:
            f.write(f'train-stg0 epoch {epoch} - auc {m0["auc"]:.4f} ap {m0["ap"]:.4f}')
            f.write('\n')
        if metric > best[0]:
            best = metric, epoch
        if epoch - best[-1] > patience:
            break
        scheduler.step()
        
        """
        #save embedding to see if it learns anything
        if (epoch-1)%50==0:
            #model.saveEmbedding(x, epoch, edge, heads, rels, weeks, days, "stg0")
            model.saveEmbedding2(epoch,task+"_seed"+str(seed)+"_stg0")
        """

    # round 2
    model = Net(
        gnn, num_feature=num_feat+2*emb_dim, n_heads=n_heads,
        num_embed=0 if gnn is None else n_hid, node_count=node_count, emb_dim=emb_dim,
        n_hidden=n_hid, num_output=1, dropout=dropout, n_layer=2
    )
    model = model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=1e-3, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=128,
        num_training_steps=best[-1]+1)

    mask_train = mask_train_origin
    y0_train = y0[mask_train]


    pbar = tqdm.tqdm(desc='train-stg1', total=best[-1])
    for epoch in range(1, best[-1]+1):
        pbar.update(1)
        train()
        m0 = test(epoch, mask_test)

        pbar.set_description(f'train-stg1 epoch {epoch} - auc {m0["auc"]:.4f} ap {m0["ap"]:.4f}')
        with open(results_file, 'a') as f:
            f.write(f'train-stg1 epoch {epoch} - auc {m0["auc"]:.4f} ap {m0["ap"]:.4f}')
            f.write('\n')

        for k, v in m0.items():
            stats[k] = v
        scheduler.step()
        """
        if (epoch-1)%50==0:
            #model.saveEmbedding(x, epoch, edge, heads, rels, weeks, days, "stg1")
            model.saveEmbedding2(epoch, task+"_seed"+str(seed)+"_stg1")
        """
    row = pd.DataFrame([stats])
    if os.path.exists(path_result):
        result = pd.read_csv(path_result)
    else:
        result = pd.DataFrame()
    result = result.append(row)
    result.to_csv(path_result, index=False)


if __name__ == '__main__':
    fire.Fire(main)