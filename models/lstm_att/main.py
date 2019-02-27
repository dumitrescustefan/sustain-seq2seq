# add package root
import os, sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn

from lstm_att.lstm import LSTMEncoderDecoderAtt

# loading data
#import util.loaders
import util.biloaders

#data_folder = os.path.join("..","..","data","cnndm","bpe","ready","cnndm.8K.bpe.model")
data_folder = os.path.join("..","..","data","roen","ready","setimes.8K.bpe")

batch_size = 32
print("Loading data ...")
#train_loader, valid_loader, test_loader, w2i, i2w = util.loaders.prepare_dataloaders(data_folder, batch_size, 1000, 5)
train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = util.biloaders.prepare_dataloaders(data_folder, batch_size, 1000, 5)
print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size {}\n".format(
    len(train_loader.dataset.X),
    len(valid_loader.dataset.X),
    len(test_loader.dataset.X),
    len(src_w2i)))


# x and y start with BOS (2), end with EOS(3), are padded with PAD (0) and unknown words are UNK (1)
# example batch
dataiter = iter(train_loader)
# x_sequence, x_pos, y_sequence, y_pos = dataiter.next() # if pos loader is used
x_sequence, y_sequence = dataiter.next()
from pprint import pprint
pprint(x_sequence[0])
print(y_sequence[0]) # ex: tensor([    2, 12728, 49279, 13516,  4576, 25888,  1453,     1,  7975, 38296, ...])


# Instantiate the model w/ hyperparams
embedding_dim = 512 #128 #10 #100
encoder_hidden_dim = 512 #256 #128 #256
decoder_hidden_dim = 512 #encoder_hidden_dim*2 # for bidirectional LSTM in the encoder
encoder_n_layers = 2
decoder_n_layers = 1
encoder_drop_prob = 0.3
decoder_drop_prob = 0.3
lr = 0.0005

net = LSTMEncoderDecoderAtt(src_w2i, src_i2w, tgt_w2i, tgt_i2w, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, encoder_n_layers, decoder_n_layers, encoder_drop_prob=encoder_drop_prob, decoder_drop_prob=decoder_drop_prob, lr = lr, model_store_path = "../../train/lstm_att")

print(net)

# train
#net.load_checkpoint("last")
net.train(train_loader, valid_loader, test_loader, batch_size, patience = 20)


# run
#net.load_checkpoint("best")
#input = [ [4,5,6,7,8,9], [9,8,7,6] ]
#output = net.run(input)
#output = net.run(valid_loader, batch_size)
#print(output)