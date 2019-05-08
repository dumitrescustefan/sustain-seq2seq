import os, sys
sys.path.insert(0, '../..')

from data.e2e.loader import loader
from models.transformer.model import Transformer
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

data_folder = "../../data/e2e"


def _print_some_examples(model, loader, seq_len):
    X, y = iter(loader).next()    
    X = X[0:seq_len]
    y = y[0:seq_len]
    if model.cuda:
        X = X.cuda()
        y = y.cuda()

    out_valid = model.forward(X, y)
    out_valid = torch.argmax(out_valid, dim=2)

    for i in range(seq_len): 
        for j in range(len(X[i])):
            token = str(X[i][j].item())
            
            if token not in tgt_i2w.keys():
                print(src_i2w['1'] + " ", end='')
            elif token == '3':
                print(src_i2w['3'], end='')
                break
            else:
                print(src_i2w[token] + " ", end='')
        print()        
        for j in range(len(y[i])):
            token = str(y[i][j].item())
            
            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'], end='')
                break
            else:
                print(tgt_i2w[token] + " ", end='')
        print()    
        for j in range(len(out_valid[i])):
            token = str(out_valid[i][j].item())
            
            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'], end='')
                break
            else:
                print(tgt_i2w[token] + " ", end='')
        print()    
        print("-"*40)


def train(model, epochs, batch_size, n_class, train_loader, valid_loader, test_loader, tgt_i2w):    
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_data = len(train_loader.dataset.X)
    n_dev_data = len(valid_loader.dataset.X)
    
    # training
    for epoch in range(epochs):
        model.train()
        print("Epochs: {}/{}\n".format(epoch+1, epochs))
        average_loss = 0
        cnt = 0
        
        t = tqdm(train_loader, desc="Epoch "+str(epoch), unit="batches")        
        for (x_batch, y_batch) in t:            
            if model.cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                
            #if batch_counter%100==0:
            #    print("Batch: {}/{}".format(batch_counter, int(n_data/batch_size)))

            optimizer.zero_grad()

            output = model.forward(x_batch, y_batch)

            loss = criterion(output.view(-1, n_class), y_batch.contiguous().view(-1))
            average_loss += loss
            
            #if batch_counter%100==0:
            #    print("Loss: {}".format(loss))
            loss.backward()
            optimizer.step()
            
            cnt+=1
            t.set_postfix(loss=average_loss.data.item()/cnt) # print loss
            #_print_some_examples(model, test_loader, 1)
            

        # deving
        model.eval()
        _print_some_examples(model, test_loader, batch_size)
        dev_accuracy = 0

        for x_dev_batch, y_dev_batch in valid_loader:
            if model.cuda:
                x_dev_batch = x_dev_batch.cuda()
                y_dev_batch = y_dev_batch.cuda()
            out_valid = model.forward(x_dev_batch, y_dev_batch).argmax(dim=2).view(-1)

            dev_accuracy += accuracy_score(y_dev_batch.view(-1).cpu(), out_valid.cpu())

        print("\nValidation Accuracy: {}".format(dev_accuracy/(n_dev_data/batch_size)))
        print("Average loss: {}\n".format(average_loss/(n_data/batch_size)))


def evaluate():
    # to do
    pass


if __name__ == "__main__":
    batch_size = 32
    min_seq_len = 10
    max_seq_len = 10000

    print("Loading data ...")
    train_loader, valid_loader, test_loader, src_w2i, src_i2w, tgt_w2i, tgt_i2w = loader(data_folder, batch_size, max_seq_len, min_seq_len)
    
    print("Loading done, train instances {}, dev instances {}, test instances {}, vocab size {}\n".format(
        len(train_loader.dataset.X),
        len(valid_loader.dataset.X),
        len(test_loader.dataset.X),
        len(tgt_i2w)))

    # train_loader.dataset.X = train_loader.dataset.X[0:500]
    # train_loader.dataset.y = train_loader.dataset.y[0:500]
    # valid_loader.dataset.X = valid_loader.dataset.X[0:500]
    # valid_loader.dataset.y = valid_loader.dataset.y[0:500]

    n_class = len(tgt_i2w)
    # number of enc/dec
    N = 1
    # embedding size
    d_model = 64
    # feed_forward size
    d_ff = 512
    # number of heads
    h = 4
    # embedding size for keys
    d_k = 16
    # embedding size for values
    d_v = 16
    transformer = Transformer(N, d_model, d_ff, h, d_k, d_v, n_class, max_seq_len)

    epochs = 10
    train(transformer, epochs, batch_size, n_class, train_loader, valid_loader, test_loader, tgt_i2w)
        
   