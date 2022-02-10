import numpy as np
import tqdm
import os
import pandas as pd

DIR_DATA = os.path.abspath(os.path.dirname(__file__)) + \
           '/../../../../data/shared/ebay-stg2'

def create_enc_dec_node(edge):
    enc = dict((e, i) for i, e in enumerate(edge['src'].unique()))
    for e in edge['dst'].unique():
        enc[e] = len(enc)
    dec = dict((v, k) for k, v in enc.items())
    return enc, dec

def get_data(y_selected_columns=('gmv_class', 'risky'), debug=False, gmv_fair=False):
    
    path_x = f'{DIR_DATA}/feat.parquet'
    path_y = f'{DIR_DATA}/label.parquet'
    path_edge = f'{DIR_DATA}/edge.parquet'
    
    x = pd.read_parquet(path_x)
    y = pd.read_parquet(path_y)
    edge = pd.read_parquet(path_edge)

    if debug:
        edge = edge.sample(n=10000)

    edge = edge.sort_values(by = ["edge_type"]) #added to conform to data format of Heterogeneous Graph Benchmark
    edge_new = pd.merge(
        edge, x[['node_id', 'ts_wk', 'ts']],
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
    print(df_tpl.shape)
    x = pd.merge(df_tpl, x, how='left', on='node_id')
    x.fillna(0., inplace=True)

    y = y.fillna(0)
    y = pd.merge(df_tpl, y, how='left', on='node_id')
    
    def apply(gmv):
        if gmv > 1000:
            return 4
        if gmv > 100:
            return 3
        if gmv > 10:
            return 2
        if gmv > 0:
            return 1
        if gmv == 0:
            return 0
        return -1

    y['gmv_class'] = y['total_gmv'].apply(apply)
    y['risky'] = (
            (y['lbl0']+y['lbl1']+y['lbl2']+
             y['lbl3']+y['lbl4']+y['lbl5'])
            > 0).astype(int)
    y.loc[y.gmv_class<0, 'risky'] = -1

    edge = pd.merge(
            edge, y[['node_id','risky']],
            left_on='src', right_on='node_id')

    y = y[list(y_selected_columns)]

    return g, x, y, g_ts, edge