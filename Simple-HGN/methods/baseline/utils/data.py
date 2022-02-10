import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
from utils.dysattools_massreg import get_data as get_data_massreg
from utils.dysattools_xfraud import get_data as get_data_xfraud

def load_data(prefix='DBLP', chronological_split=1):
    from scripts.data_loader import data_loader
    dl = data_loader('../../data/'+prefix)
    
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    
    if chronological_split:
        # chronological train-val split
        print("chronological train-val split", flush=True)
        if prefix == "massreg":
            print("MassReg dataset")
            g, x, y, g_ts, edge = get_data_massreg()
            mask_train = (y['gmv_class']>=0) & (x['phase'] == 'TRAIN') & (x['ts'] <= 58)
            train_idx = np.nonzero(np.array(mask_train))[0]
            mask_val = (y['gmv_class']>=0) & (x['phase'] == 'TRAIN') & (x['ts'] > 58)
            val_idx = np.nonzero(np.array(mask_val))[0]
        elif prefix == "xfraud_txn" or prefix == "xfraud_account":
            print("xFraud datasets")
            g, x, y, g_ts, edge = get_data_xfraud()
            # we use different week threshold to have a split of 70-15-15 in every case
            if prefix =="xfraud_txn":
                wk_thresh = 3
            else:
                wk_thresh = 2
            mask_train = (x['phase'] == 'TRAIN') & (x['ts_wk'] <= wk_thresh)
            train_idx = np.nonzero(np.array(mask_train))[0]
            mask_val = (x['phase'] == 'TRAIN') & (x['ts_wk'] > wk_thresh)
            val_idx = np.nonzero(np.array(mask_val))[0]
        else:
            raise NotImplementedError('unknown dataset %s' % prefix)
    
    else:
        # random train-val split
        print("random train-val split", flush=True)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
    
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl
