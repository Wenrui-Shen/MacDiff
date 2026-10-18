import json
import sys
import yaml
import numpy as np
import os
from datetime import datetime
import random
import string
import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
from feeder.feeder_ntu import Feeder

ntu_bone = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))

def visualize_ntu_torch(data_torch, name=None, label=None, save_path=None, interval=4):
    data_numpy = data_torch.detach().cpu().numpy()
    visualize_ntu(data_numpy, name, label, save_path, interval=interval)

def visualize_ntu(data_numpy, name=None, label=None, save_path=None, interval=4):
    if save_path is None:
        formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f'./visualize/results_{formatted_datetime}'
    os.makedirs(save_path, exist_ok=True)
    '''
    data_numpy: T * V * C
    '''
    if len(data_numpy.shape) == 4: data_numpy = data_numpy[...,0]
    if data_numpy.shape[0] == 3: data_numpy = data_numpy.transpose(1,2,0)
    T, V, C = data_numpy.shape

    plt.close()
    fig = plt.figure(figsize=(15,15), dpi=800)
    fig.patch.set_facecolor('white')

    assert 120 % interval == 0
    for f in range(0, 120//interval):
        idx = interval * f
        datum = data_numpy[interval * f]
        # datum = data[1][1][:,f].numpy()
        # ax = Axes3D(fig)
        ax = fig.add_subplot(5,10,f+1,projection='3d')
        ax.patch.set_facecolor('white')
        links = [(3, 2), (2, 20), (20, 1), (1, 0), (0, 12), (0, 16), (12, 13), (13, 14),
                (14, 15), (16, 17), (17, 18), (18, 19), (20, 4), (4, 5), (5, 6), (6, 7), (20, 8), (8, 9), (9, 10), (10, 11)]
        
        x = datum[:,0]
        y = datum[:,1]
        z = datum[:,2]
        
        for link in links:
            i = link[0]
            j = link[1]
            ax.plot([x[i],x[j]], [y[i],y[j]], [z[i],z[j]], c='steelblue', linewidth=1.0)

        ax.scatter(x, y, z, c='darkblue', s=2)

        ax.text(0.2,-1.4,0,f"t = {idx}")
        
        ax.set_xlim(-0.3,0.3)
        ax.set_ylim(-1,0.2)
        ax.set_zlim(-1,0)

        ax.view_init(elev=-90, azim=90)
        plt.axis('off')

    plt.show()
    if name is None: name = random.randint(10**10, 9*10**10)
    if label is None: label = 'nolabel'
    plt.savefig(f'{save_path}/name_{name}_label_{label}.png')



if __name__ == '__main__':
    print('Loading NTU dataset...')

    ntu_feeder = Feeder(
        data_path='data/ntu/NTU60_XSub.npz',
        window_size=120,
    )
    print('Visualizing images...')
    for i in range(10):
        idx = random.randint(0,len(ntu_feeder.data))
        data_numpy = ntu_feeder.data[idx]
        label = ntu_feeder.label[idx]
        sys.stdout.flush()
        visualize_ntu(data_numpy, name=idx, label=label, save_path='tmp', interval=12)