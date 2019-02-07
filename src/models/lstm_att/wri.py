# this file is for playing with SummaryWriter, no need for anything else

from pprint import pprint
import time, io

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

writer = SummaryWriter('/work/tmp/2')
x = torch.randn(512,64)
print(x.size())
#print(x)



"""
#fig = plt.figure(figsize=(10,15))
fig = plt.figure()
#plt.plot(x.numpy())
#ax=plt.subplot(111)
#sns.heatmap(x.numpy(),ax=ax)
sns.heatmap(x.numpy(), cmap="YlGnBu")

self.writer.add_figure('matplotlib', fig, global_step=0)
"""


def plot_tensor(tensor):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.plot(tensor)        
    return fig


def show_tensor(x, prediction=None, source=None):
    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(x,
                #xticklabels=prediction.split(),
                #yticklabels=source.split(),
                #linewidths=.05,
                cmap="rainbow")
    #plt.ylabel('Source (German)')
    #plt.xlabel('Prediction (English)')
    #plt.xticks(rotation=60)
    #plt.yticks(rotation=0)
   
    plt.tight_layout()
    
    return fig
        
   
#fig.subplots_adjust(top=0.93)
#fig.suptitle('Wine Attributes Correlation Heatmap',                       fontsize=14,                       fontweight='bold')
f = show_tensor(x.numpy())
writer.add_figure('asdasdaw', f)
f.savefig('foo.png')


x = torch.randn(512,256)
f = show_tensor(x.numpy())
writer.add_figure('asdasdaw', f)
f.savefig('foo2.png')

writer.close()