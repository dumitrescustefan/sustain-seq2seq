# add package root
import os, sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn

from lstm.lstm import LSTMEncoderDecoder

# loading data
import loaders.loaders
data_foler = os.path.join("..","..","train","transformer")
batch_size = 64
print("Loading data ...")
train_loader, valid_loader, test_loader, w2i, i2w = loaders.loaders.prepare_dataloaders(data_foler, batch_size)
print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size {}\n".format(
    len(train_loader.dataset.X),
    len(valid_loader.dataset.X),
    len(test_loader.dataset.X),
    len(w2i)))

"""
# example batch
dataiter = iter(train_loader)
# x_sequence, x_pos, y_sequence, y_pos = dataiter.next() # if pos loader is used
x_sequence, y_sequence = dataiter.next()
from pprint import pprint
pprint(y_sequence[0]) # ex: tensor([    2, 12728, 49279, 13516,  4576, 25888,  1453,     1,  7975, 38296, ...])
"""

# Instantiate the model w/ hyperparams
embedding_dim = 100
encoder_hidden_dim = 256
decoder_hidden_dim = 256
encoder_n_layers = 2
decoder_n_layers = 2
encoder_drop_prob=0.3
decoder_drop_prob=0.3
lr = 0.001

net = LSTMEncoderDecoder(w2i, i2w, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_n_layers, decoder_n_layers, encoder_drop_prob=encoder_drop_prob, decoder_drop_prob=decoder_drop_prob, lr = lr)
print(net)

net.train(train_loader, valid_loader, test_loader, batch_size)