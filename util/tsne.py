import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import sys
import os
import numpy as np

import torch.distributed as dist

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_embedding_2d(X, y, title=None, save_path=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    color = plt.cm.Set3(y)
    #print(y, color)
    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=color[i],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/tsne.png')

class tsne_tool:
    def __init__(self, num_class=60, dim_feat=256, num_visualize=50):
        self.num_class = num_class
        self.dim_feat = dim_feat
        self.M = num_visualize * num_class
        self.feats = []
        self.labels = []
        #self.selected_labels = [i*6 for i in range(10)]

    def update(self, feat, label):
        '''
        feat: [N, C]
        label: [N]
        '''
        N, C = feat.shape
        assert C == self.dim_feat
        feat = feat.detach()
        label = label.detach()
        for i in range(N):
            #if label[i].item() in self.selected_labels:
            self.feats.append(feat[i:i+1])
            self.labels.append(label[i:i+1])

    def visualize(self, save_path):
        print('Visualizing t-SNE...')
        print('feat len', len(self.feats))
        print('label len', len(self.labels))
        sys.stdout.flush()
        tsne = TSNE(n_components=2, random_state=0)
        feats = torch.cat(self.feats, dim=0)[:self.M].cpu().numpy()
        y = torch.cat(self.labels, dim=0)[:self.M].cpu().numpy()
        feats = tsne.fit_transform(feats)
        plot_embedding_2d(X=feats, y=y, title="t-SNE 2D", save_path=save_path)
        print('Finished t-SNE.')


            


    








