import numpy as np
import tqdm
import os
import pandas as pd

DIR_DATA = os.path.abspath(os.path.dirname(__file__)) + \
           '/../../../../data/shared/ebay-xfraud'

def create_enc_dec_node(edge):
    enc = dict((e, i) for i, e in enumerate(edge['src'].unique()))
    for e in edge['dst'].unique():
        enc[e] = len(enc)
    dec = dict((v, k) for k, v in enc.items())
    return enc, dec

def get_data(prefix, debug=False, gmv_fair=False):
    
    path_x = f'{DIR_DATA}/' + prefix + '/feat.parquet'
    path_y = f'{DIR_DATA}/' + prefix + '/label.parquet'
    path_edge = f'{DIR_DATA}/' + prefix + '/edge.parquet'
    
    x = pd.read_parquet(path_x)
    y = pd.read_parquet(path_y)
    edge = pd.read_parquet(path_edge)

    if debug:
        edge = edge.sample(n=10000)

    edge = edge.sort_values(by = ["edge_type"]) #added to conform to data format of Heterogeneous Graph Benchmark
    edge_new = pd.merge(
        edge, x[['node_id']],
        left_on='src', right_on='node_id')
    enc, dec = create_enc_dec_node(edge)
    edge_new['dst_origin'] = edge_new['dst']
    x['node_id'] = x['node_id'].map(enc)
    y['node_id'] = y['node_id'].map(enc)

    edge=edge_new
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

    edge = pd.merge(
            edge, y[['node_id','risky']],
            left_on='src', right_on='node_id')

    return g, x, y, g_ts, edge