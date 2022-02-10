import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import glob
import os
import torch
from train_mass_reg import get_data

def plotEmbedding(name=''):
    
    if not os.path.exists('tsne_viz'):
        os.makedirs('tsne_viz')
    dysat = True
    gmv_fair = False
    g, x, y, g_ts, edge, _ = get_data(ts_snapshot=dysat, gmv_fair=gmv_fair, debug=False, get_node_type=False)

    files = glob.glob('stg*.pt') # change path here if needed
    for file in files:
        plt.figure()
        emb = torch.load(file)
        X = emb[:10000].cpu().detach().numpy()
        X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=False).fit_transform(X)
        y_lbl = y['risky'].values
        y_lbl = y_lbl[:10000]
        num_labels=2
        current_palette = sns.color_palette("colorblind")
        current_palette=current_palette[0:num_labels]
        legend = ["0", "1"]
        colors = current_palette
        for color, i, target_name in zip(colors, range(num_labels), np.unique(y_lbl)):
            plt.scatter(X_emb[y_lbl == target_name, 0], X_emb[y_lbl == target_name, 1],
                              color=color, lw=2, label=target_name, alpha=0.5, s=0.7)
            plt.legend(legend, loc="best", shadow=False, scatterpoints=1)
            plt.savefig("tsne_viz/tsne-"+file[:-3]+".png")
        plt.close()
        

if __name__ == "__main__":
    plotEmbedding()